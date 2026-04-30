#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import statsmodels.stats.multitest as smt
from pandas.plotting._matplotlib.style import get_standard_colors





analysis_folder = os.path.join("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data")



"read the ICD to phecode mapping file"
icd_to_pheCode_map_10 = pd.read_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/Phecode_map_v1_2_icd10cm_beta 3.csv)
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
# df_baby_cond.to_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/PheWAS_babiesOver15_withCBC/data/babies_condirions.csv", index=False)

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
## remove age with negative values
# df_baby_cond = df_baby_cond[df_baby_cond['age_at_cond']>=0]
## just conditions over 10 or 12 or 15 weeks
df_baby_cond = df_baby_cond[df_baby_cond['age_at_cond']>=14]

## remove records for age more than 14 weeks
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

""" keep conditions that have at least been recorded twice for cases"""
# df_baby_cond_cases = df_baby_cond[df_baby_cond['label']==1]
# condition_cases = df_baby_cond_cases.groupby(['person_id','icd10cm']).size().reset_index()
# condition_cases = condition_cases.rename(columns={0:'count_conditions'})
# condition_cases = condition_cases[condition_cases['count_conditions']>=2]
# cond_of_int = condition_cases['icd10cm']
# df_baby_cond = df_baby_cond[df_baby_cond['icd10cm'].isin(cond_of_int)]
""" map conditions to PheCode"""
# df_baby_cond_phecode = pd.merge(df_baby_cond,icd_to_pheCode_map,how='inner',on='icd10cm')
df_baby_cond_phecode = pd.merge(df_baby_cond,icd_to_pheCode_map,how='left', on='icd').drop_duplicates()
### df_baby_cond_phecode['phecode_str'] = df_baby_cond_phecode['phecode_str'].fillna('No_map')



""" separating cases """
df_baby_cond_cases = df_baby_cond_phecode[df_baby_cond_phecode['label']==1]
df_baby_cond_cont = df_baby_cond_phecode[df_baby_cond_phecode['label']==0]

phecode_count_cases = df_baby_cond_cases.groupby(['person_id','phecode_str'])['condition_DATE'].agg('first').reset_index()
babies_withPhecode_count_cases = phecode_count_cases.groupby(['phecode_str']).count().reset_index()

phecode_count_cont = df_baby_cond_cont.groupby(['person_id','phecode_str'])['condition_DATE'].agg('first').reset_index()
babies_withPhecode_count_cont = phecode_count_cont.groupby(['phecode_str']).count().reset_index()

## keep phecode with 20 or more people recorded with in each class
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




df_baby_dummy = pd.get_dummies(df_baby_cond_phecode_intPheCode, columns=['phecode_str'])
df_baby_dummy.columns = [col.replace('phecode_str_', '') for col in df_baby_dummy.columns]
df_baby_dummy['exclude_range'] = df_baby_dummy['exclude_range'].fillna("No_map")
df_baby_dummy['exclude_name'] = df_baby_dummy['exclude_name'].fillna("No_map")
df_baby_dummy = df_baby_dummy.drop_duplicates()
df_baby_dummy_agg = df_baby_dummy.groupby('person_id').agg('max').reset_index()
df_baby_dummy_agg.columns = [col.replace('/', '.') for col in df_baby_dummy.columns]

X = df_baby_dummy_agg.drop(columns=['age_at_cond','label','person_id','phecode','condition_DATE','icd','icd_str','condition_start_DATETIME','birth_DATETIME','exclude_name','exclude_range'])

variables = X.columns
tmp = pd.DataFrame()
result = pd.DataFrame()
crosstab_folder = "/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/crosstab"

for loc,i in enumerate(variables):
    X_ind = df_baby_dummy_agg[[i]]
    tmp_crstb = pd.crosstab(index=df_baby_dummy_agg['label'], columns=X_ind[i])
    tmp_crstb.to_csv(os.path.join(crosstab_folder, tmp_crstb.columns.name+".csv"))
    chi2, pvalue, _, _  = chi2_contingency (tmp_crstb)
    result['disease'], result['chi2'], result['pvalue'] = [i],[chi2], [pvalue] 
    tmp = pd.concat([tmp,result], axis=0)


# # tmp.to_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/chi2_diseases_deadexclude.csv", index=False)

# df = pd.read_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/chi2_diseases_deadexclude_aggr_30samplesThresh.csv")

# df.columns = [col.replace('phecode_str_', '') for col in df.columns]
# df.columns = df.apply(lambda x: ''.join(filter(str.isalnum, str(x.name))))

# just for p_value
df = tmp
df_tmp = df[['disease','pvalue']]
df_tmp = df_tmp.set_index('disease')

method='fdr_bh'
# method = 'bonferroni'
pvals = smt.multipletests (df_tmp['pvalue'].values, method=method)
df_tmp['p_value_adj'] = pvals[1]
df_tmp['include'] = pvals[0]
# df_tmp = df_tmp[df_tmp['include']].drop(columns='include')

df_tmp['p_value_adj'] = -np.log10(df_tmp['p_value_adj'].values)
# df_tmp.to_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/chi2_diseases_deadexclude_icd9include_benjaminiH_correction.csv")
# m = df_tmp.loc[df_tmp['p_value_adj'] != np.inf, 'p_value_adj'].max()
# df_tmp['p_value_adj'].replace(np.inf,m,inplace=True)

## read the ICD to phecode mapping file
"read the ICD to phecode mapping file"
icd_to_pheCode_map_10 = pd.read_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/PheWAS_babiesOver15_withCBC/data/Phecode_map_v1_2_icd10cm_beta 3.csv", encoding="ISO-8859-1")

# df_ref = icd_to_pheCode_map.replace(r'[^0-9a-zA-Z]', '', regex=True)
df_ref = icd_to_pheCode_map_10
df_ref['phecode_str'] = df_ref['phecode_str'].str.replace('/','.')
## adding the Phenotype group based on PheCode
df_merge_group = df_tmp.merge(df_ref, how='left', left_on=df_tmp.index, right_on = df_ref['phecode_str']).drop(columns=['icd10cm','icd10cm_str','exclude_range','leaf','rollup'])
df_merge_group = df_merge_group.drop_duplicates()
df_merge_group['exclude_name'] = df_merge_group['exclude_name'].fillna(value='No Mapping')
groups = df_merge_group.groupby('exclude_name')

# Plot
colors = get_standard_colors(len(groups), color_type='random')

fig, ax = plt.subplots(1,1,figsize=(20, 10))
ax.set_prop_cycle('color', colors)
ax.margins(0.05)
for name, group in groups:
    ax.plot(group['key_0'], group['p_value_adj'], marker='o', linestyle='', ms=12, label=group['exclude_name'].unique())
    id_max = group['p_value_adj'].idxmax()
    group_df = pd.DataFrame(group['p_value_adj'])
    id_int = group[group['include']==True].sort_values(by='p_value_adj', ascending=False)
    id_int_top = id_int.nlargest(3,'p_value_adj')
    # if id_max in id_int:
    #     ax.text(group['key_0'].loc[id_max],group['p_value_adj'].loc[id_max],s=group['key_0'].loc[id_max], fontsize = 'small', rotation=45)
    for id in id_int_top.index:
        ax.text(group['key_0'].loc[id],group['p_value_adj'].loc[id],s=group['key_0'].loc[id], fontsize = 'medium', rotation=35)

# ax.axhline(y=0.023, color='b', linestyle='--') #For AUPRC
ax.axhline(y=-np.log10(0.05), color='b', linestyle='--', label='alpha=0.05') #For p_value
ax.set_xlabel("Phecode categories of different conditions" ,fontsize=12)
ax.set_ylabel("$-log_{10} (p-value)$", fontsize=12)

# ax.get_xaxis().set_visible(False)
ax.get_xaxis().set_ticks([])
# plt.xticks(rotation = 45)
ax.legend(numpoints=1, loc='upper left')
# plt.legend(loc=(1.04,0))
plt.tight_layout()

plt.show(block=False)
fig.savefig(os.path.join(analysis_folder,'chi2_p_diseases_deadexl.pdf'))

