import pandas as pd
from google.cloud import bigquery
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np
import matplotlib.pyplot as plt
from torch import rand 
from sklearn import metrics
from sklearn.metrics import auc,precision_recall_curve,roc_auc_score,confusion_matrix
import seaborn as sns
from numpy import argmax
from scipy.stats import mannwhitneyu
import json
from statsmodels.regression.linear_model import OLS
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon, chisquare, chi2_contingency
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from scipy.stats import mannwhitneyu
import scipy.stats as stats
import statsmodels.stats.multitest as smt
import statsmodels.formula.api as smf





analysis_folder = os.path.join("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data")



"read the ICD to phecode mapping file"
icd_to_pheCode_map_10 = pd.read_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/Phecode_map_v1_2_icd10cm_beta 3.csv", encoding="ISO-8859-1")
icd_to_pheCode_map_09 = pd.read_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/phecode_icd9_rolled.csv", encoding="ISO-8859-1")
icd_to_pheCode_map_10.rename(columns={'icd10cm' : 'icd', 'icd10cm_str':'icd_str'}, inplace=True)
icd_to_pheCode_map_10.columns = map(str.lower, icd_to_pheCode_map_10.columns)
icd_to_pheCode_map_09.columns = map(str.lower, icd_to_pheCode_map_09.columns)

icd_to_pheCode_map_09.rename(columns={'icd9' : 'icd', 'icd9 string':'icd_str'}, inplace=True)
icd_to_pheCode_map_09.rename(columns={'phenotype':'phecode_str', 'excl. phecodes':'exclude_range', 'excl. phenotypes':'exclude_name'},inplace=True)
icd_to_pheCode_map_09.drop(columns=['leaf','rollup','ignore bool'], inplace=True)
icd_to_pheCode_map_10.drop(columns=['leaf','rollup'], inplace=True)

icd_to_pheCode_map = pd.concat([icd_to_pheCode_map_10, icd_to_pheCode_map_09], ignore_index=True)


""" read the conditions for all babies from birth up to 26 weeks old (FROM GCP) """
## initialize the environment and client info for BigQuery
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'/home/npayrov/.config/gcloud/application_default_credentials.json' 

client = bigquery.Client(project='som-nero-phi-naghaeep-starr')

## create the query (since the table is already prepared on GCP, we just read the table in)
Query_conditions = """
SELECT * FROM `som-nero-phi-naghaeep-starr.neelu_explore.conditions_babies_all`
"""
# df_baby_cond = client.query(Query_conditions).to_dataframe()
# df_baby_cond.to_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/babies_conditions.csv", index=False)

## read in death info
Query_deaths = """
SELECT * FROM `som-nero-phi-naghaeep-starr.neelu_explore.Baby_death_info`
"""
df_baby_deaths = client.query(Query_deaths).to_dataframe()
df_baby_deaths['age_at_death'] = ((pd.to_datetime(df_baby_deaths['death_DATE'])-pd.to_datetime(df_baby_deaths['birth_DATETIME'])).dt.days)/7
df_baby_deaths = df_baby_deaths[df_baby_deaths['age_at_death']<=26]
df_baby_deaths_ids = df_baby_deaths.person_id
## read in saved baby condition info
df_baby_cond = pd.read_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/babies_conditions.csv")

## remove babies who died before 6 months of age
df_baby_cond = df_baby_cond[~df_baby_cond['person_id'].isin(df_baby_deaths_ids)]
df_baby_cond['condition_start_DATETIME'] = pd.to_datetime(df_baby_cond['condition_start_DATETIME'])
df_baby_cond['condition_DATE'] = df_baby_cond['condition_start_DATETIME'].dt.date
df_baby_cond = df_baby_cond.sort_values(by=['person_id', 'condition_source_value','condition_start_DATETIME'])
df_baby_cond = df_baby_cond.groupby(['person_id','condition_source_value','condition_DATE']).agg('first').reset_index()
df_baby_cond['age_at_cond'] = ((pd.to_datetime(df_baby_cond['condition_start_DATETIME'])-pd.to_datetime(df_baby_cond['birth_DATETIME'])).dt.days)/7

## just conditions over 14 weeks
df_baby_cond = df_baby_cond[df_baby_cond['age_at_cond']>=14]

## remove records for age more than 26 weeks
df_baby_cond = df_baby_cond[df_baby_cond['age_at_cond']<=26]
df_baby_cond.rename(columns={'condition_source_value' : 'icd'}, inplace=True)
df_baby_cond.drop_duplicates(inplace=True)


""" add labels """
labels = pd.read_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/babies_over15weeks_CBC.csv")
df_baby_cond['label'] = (df_baby_cond['person_id'].isin(labels['person_id'])).astype('int')


""" keep conditions that have been recorded at least twice in two separate days """
disease_icd_count = df_baby_cond.groupby(['person_id','icd'])['condition_DATE'].count().reset_index()
disease_icd_count_tmp = disease_icd_count[disease_icd_count['condition_DATE']>=2]
icd_int = disease_icd_count_tmp['icd'].unique()
df_baby_cond  = df_baby_cond[df_baby_cond['icd'].isin(icd_int)]


""" map conditions to PheCode"""
df_baby_cond_phecode = pd.merge(df_baby_cond,icd_to_pheCode_map,how='left', on='icd').drop_duplicates()

""" separating cases """
df_baby_cond_cases = df_baby_cond_phecode[df_baby_cond_phecode['label']==1]
df_baby_cond_cont = df_baby_cond_phecode[df_baby_cond_phecode['label']==0]

phecode_count_cases = df_baby_cond_cases.groupby(['person_id','phecode_str'])['condition_DATE'].agg('first').reset_index()
babies_withPhecode_count_cases = phecode_count_cases.groupby(['phecode_str']).count().reset_index()

phecode_count_cont = df_baby_cond_cont.groupby(['person_id','phecode_str'])['condition_DATE'].agg('first').reset_index()
babies_withPhecode_count_cont = phecode_count_cont.groupby(['phecode_str']).count().reset_index()

## keep phecode with 30 or more people recorded with in each class
babies_withPhecode_count_cases = babies_withPhecode_count_cases[babies_withPhecode_count_cases['person_id']>=15]
phecode_to_include_cases = babies_withPhecode_count_cases['phecode_str']

babies_withPhecode_count_cont = babies_withPhecode_count_cont[babies_withPhecode_count_cont['person_id']>=15]
phecode_to_include_cont = babies_withPhecode_count_cont['phecode_str']

phecode_to_include = phecode_to_include_cases[phecode_to_include_cases.isin(phecode_to_include_cont)]


## only keeping phecodes of interest
df_baby_cond_phecode_intPheCode = df_baby_cond_phecode[df_baby_cond_phecode['phecode_str'].isin(phecode_to_include)].reset_index().drop(columns=['index','label'])
cases_ids = pd.DataFrame(phecode_count_cases['person_id'].unique())
df_baby_cond_phecode_intPheCode['label'] = (df_baby_cond_phecode_intPheCode['person_id'].isin(cases_ids[0])).astype('int')
# df_baby_cond_phecode_intPheCode_agg = df_baby_cond_phecode_intPheCode.groupby(['person_id', 'icd','phecode']).agg('first').reset_index()
baby_ga = pd.read_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/PheWAS_babiesOver15_withCBC/data/Babies_GA.csv")
df_baby_cond_phecode_intPheCode['GA'] = pd.merge(df_baby_cond_phecode_intPheCode,baby_ga,how='left', on='person_id').Total_GA_Days.astype(float)
df_baby_cond_phecode_intPheCode['GA'] = df_baby_cond_phecode_intPheCode['GA'] .fillna(df_baby_cond_phecode_intPheCode['GA'].mean())

df_baby_dummy = pd.get_dummies(df_baby_cond_phecode_intPheCode, columns=['phecode_str'])
df_baby_dummy.columns = [col.replace('phecode_str_', '') for col in df_baby_dummy.columns]
df_baby_dummy['exclude_range'] = df_baby_dummy['exclude_range'].fillna("No_map")
df_baby_dummy['exclude_name'] = df_baby_dummy['exclude_name'].fillna("No_map")
df_baby_dummy = df_baby_dummy.drop_duplicates()
df_baby_dummy_agg = df_baby_dummy.groupby('person_id').agg('max').reset_index()



### --------------------- beginning of bivariate analysis 
##df_baby_dummy = df_baby_dummy.groupby('person_id').agg('max').reset_index()
y = df_baby_dummy_agg['label']
X = df_baby_dummy_agg.drop(columns=['age_at_cond','label','person_id','phecode','condition_DATE','icd','icd_str','condition_start_DATETIME','birth_DATETIME','exclude_name','exclude_range'])


variables = X.columns
tmp = pd.DataFrame()

for loc,i in enumerate(variables):

    if (i!='GA'): 
        X_train = df_baby_dummy_agg[[i, 'GA']]
        y_train  = df_baby_dummy_agg[['label']]
        model = sm.Logit(y_train,X_train).fit(method='bfgs')
        params = model.params
        conf = model.conf_int()
        conf['coef'] = params
        conf.columns=['5%', '95%', 'coeff']
        conf['Odds'] = np.exp(conf['coeff'] )
        conf['5%'] = np.exp(conf['5%'])
        conf['95%'] = np.exp(conf['95%'])
        conf['odds_pvalue_regression'] = model.pvalues
        y_score = model.predict(X_train, 'prob')
        tmp_crstb = pd.crosstab(index=df_baby_dummy_agg['label'], columns=df_baby_dummy_agg[i])
        chi2, chi2_pvalue, _, _  = chi2_contingency (tmp_crstb)
        conf = conf.drop('GA')
        # conf[i], conf['chi2'], conf['chi2_pvalue'] = [i],[chi2], [-np.log10(chi2_pvalue)] 
        conf['chi2'], conf['chi2_pvalue'] = [chi2], [-np.log10(chi2_pvalue)] 

        oddsratio_fisher, pvalue_fisher = stats.fisher_exact(tmp_crstb)
        conf['fisher_pvalue'] = -np.log10(pvalue_fisher)
        conf['odds_fisher'] = oddsratio_fisher
        conf['n_Dis_positive'] = tmp_crstb[1][1]
        conf['n_NoDis_positive'] = tmp_crstb[0][1]
        conf['n_Dis_negative'] = tmp_crstb[1][0]
        conf['n_NoDis_negative'] = tmp_crstb[0][0]
        conf['total_label_negative'] = conf['n_Dis_negative'] + conf['n_NoDis_negative']
        conf['total_label_positive'] = conf['n_Dis_positive'] + conf['n_NoDis_positive']
        positive_labels = y_score[y==1]
        negative_labels = y_score[y==0]
        # disease = X_ind
        # label = df_baby_dummy_agg['label']

        _, p_value = mannwhitneyu(positive_labels, negative_labels)
        # _, p_value = mannwhitneyu(disease, label)

        conf['U_pvalue'] = -np.log10(p_value)
        tmp = pd.concat([tmp,conf], axis=0)

method = 'fdr_bh'
pvals = smt.multipletests (tmp['odds_pvalue_regression'].values, method=method)
tmp['p_value_regression_adj'] = pvals[1]
tmp['include'] = pvals[0]




# # significant_p = -np.log10(0.001)
# # tmp_odds_pvalue = tmp[tmp['oddspvalue']>=significant_p]
# # tmp_U_pvalues = tmp[tmp['U_pvalue']>=significant_p]
# # tmp_both_pvalues = tmp[(tmp['U_pvalue']>=significant_p) & (tmp['oddspvalue']>=significant_p)]

tmp.to_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/disease_phecode_bivariate_oddsratio_aggr_1031_14weeksto26_all_notadjusted_GAincluded.csv")

### --------------------- end of bivariate analysis 

