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
import statsmodels.stats.multitest as smt

analysis_folder = os.path.join("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data")


# """ read the medications with ATC(n) for all babies"""
# # ## initialize the environment and client info for BigQuery
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'/home/npayrov/.config/gcloud/application_default_credentials.json' 
# client = bigquery.Client(project='som-nero-phi-naghaeep-starr')

# ## create the query (since the table is already prepared on GCP, we just read the table in)
# Query_medication = """
# SELECT * FROM `som-nero-phi-naghaeep-starr.neelu_explore.medication_ATC4_babies`
# """
# df_baby_med = client.query(Query_medication).to_dataframe()

# ## convert to parquet
# df_baby_med.to_parquet("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/PheWAS_babiesOver15_withCBC/data/babies_medication_atc4.gzip", compression='gzip',index=False)

""" read in the saved data """
df_baby_med = pd.read_parquet("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/babies_medication_atc3.gzip")

""" read the conditions for all babies from birth up to 26 weeks old (FROM GCP) """
## initialize the environment and client info for BigQuery
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'/home/npayrov/.config/gcloud/application_default_credentials.json' 
client = bigquery.Client(project='som-nero-phi-naghaeep-starr')

## read in new (RxNorm FLT data)
Query_med = """
SELECT * FROM `som-nero-phi-naghaeep-starr.neelu_explore.medication_ATC3_babies_FLTRxNorm`
"""

## read in death info
Query_deaths = """
SELECT * FROM `som-nero-phi-naghaeep-starr.neelu_explore.Baby_death_info`
"""
df_baby_med = client.query(Query_med).to_dataframe()
df_baby_deaths = client.query(Query_deaths).to_dataframe()
df_baby_deaths['age_at_death'] = ((pd.to_datetime(df_baby_deaths['death_DATE'])-pd.to_datetime(df_baby_deaths['birth_DATETIME'])).dt.days)/7
df_baby_deaths = df_baby_deaths[df_baby_deaths['age_at_death']<=26]
df_baby_deaths_ids = df_baby_deaths.person_id

""" remove dead babies """
df_baby_med = df_baby_med[~df_baby_med['person_id'].isin(df_baby_deaths_ids)]

""" only keeping records between 0 and n weeks of age """
df_baby_med['drug_exposure_start_DATETIME'] = pd.to_datetime(df_baby_med['drug_exposure_start_DATETIME'])
df_baby_med['drug_exposure_start_DATETIME'] = df_baby_med['drug_exposure_start_DATETIME'].dt.date
df_baby_med = df_baby_med.sort_values(['person_id','drug_name','drug_exposure_start_DATETIME'],ascending=True).groupby(['person_id','drug_name']).agg('first').reset_index()
df_baby_med['age_at_med'] = ((pd.to_datetime(df_baby_med['drug_exposure_start_DATETIME'])-pd.to_datetime(df_baby_med['birth_DATETIME'])).dt.days)/7

## just conditions over 14
df_baby_med = df_baby_med[df_baby_med['age_at_med']>=14]
## remove records for age more than 26 weeks
df_baby_med = df_baby_med[df_baby_med['age_at_med']<=26]

""" add labels """
labels = pd.read_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/babies_over15weeks_CBC.csv")
df_baby_med['label'] = (df_baby_med['person_id'].isin(labels['person_id'])).astype('int')


""" separating cases """
df_baby_cases = df_baby_med[df_baby_med['label']==1]
df_baby_controls = df_baby_med[df_baby_med['label']==0]
drugname_count = df_baby_cases.groupby(['person_id','ATC_concept'])['drug_exposure_start_DATETIME'].count().reset_index()
babies_withDrugName_count = drugname_count.groupby(['ATC_concept']).count().reset_index()

drugname_count_cont = df_baby_controls.groupby(['person_id','ATC_concept'])['drug_exposure_start_DATETIME'].count().reset_index()
babies_withDrugName_count_cont = drugname_count_cont.groupby(['ATC_concept']).count().reset_index()

## keep drug_names with 15 or more people prescribed with 
babies_withDrugName_count = babies_withDrugName_count[babies_withDrugName_count['person_id']>=15]
babies_withDrugName_count_cont = babies_withDrugName_count_cont[babies_withDrugName_count_cont['person_id']>=15]

drugNames_to_include = babies_withDrugName_count[babies_withDrugName_count['ATC_concept'].isin(babies_withDrugName_count_cont['ATC_concept'])].ATC_concept

## only keeping drugs of interest
df_baby_medication_int = df_baby_med[df_baby_med['ATC_concept'].isin(drugNames_to_include)].reset_index().drop(columns=['index','label'])
cases_ids = pd.DataFrame(drugname_count['person_id'].unique())
df_baby_medication_int['label'] = (df_baby_medication_int['person_id'].isin(cases_ids[0])).astype('int')
df_baby_medication_int_agg = df_baby_medication_int.groupby(['person_id', 'drug_name','ATC_concept']).agg('first').reset_index()

baby_ga = pd.read_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/PheWAS_babiesOver15_withCBC/data/Babies_GA.csv")
df_baby_medication_int_agg['GA'] = pd.merge(df_baby_medication_int_agg,baby_ga,how='left', on='person_id').Total_GA_Days.astype(float)
df_baby_medication_int_agg['GA'] = df_baby_medication_int_agg['GA'] .fillna(df_baby_medication_int_agg['GA'].mean())


df_baby_dummy = pd.get_dummies(df_baby_medication_int_agg, columns=['ATC_concept'],prefix='', prefix_sep='')
X = df_baby_dummy.drop(columns=['drug_exposure_start_DATETIME','age_at_med','birth_DATETIME','drug_name','drug_concept_id','ancestor_concept_id'])
X = X.drop_duplicates()
X = X.groupby('person_id').agg('max').reset_index()
df_vars = X.drop(columns=['label','person_id'])
y = X['label']

# # # 10 fold cross validation of modeling with all features (ATC grouping)
# OR = pd.DataFrame()
# AUPRC_all = pd.DataFrame()
# AUROC_all = pd.DataFrame()
# noSkill = pd.DataFrame()

# X = df_vars
# X.columns = X.apply(lambda x: ''.join(filter(str.isalnum, str(x.name))))


# def write_json_c(path, data, indent=4):
#     with open(path, 'w') as file: 
#         json.dump(data, file, indent=indent) 


# kf = StratifiedKFold(n_splits=10,  shuffle=True, random_state=42)

# splits = kf.split(X,y)

# folds = 0
# y_real = []
# y_proba = []
# fpr_app = []
# tpr_app = []
# X_train_app = []
# X_test_app = []
# y_train_app = []
# y_test_app = []
# model_app = []
# AUPRC_app = []
# AUROC_app = []
# odds_ratio = []

# fig1, axes = plt.subplots(1,2, figsize=(15, 7))
# sns.set()


        

# dict_perf = {"fold":folds, 
# "AUPRC":[], "AUROC":[],"Odds":[], "noSkill":[], "nCases":[], "nControls":[], "p_value":[]
# }
# df_coef = []
# pvalue_logit = []


# for i, (train_index, test_index) in enumerate(splits):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#     # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.3)
#     nCases = y_test[y_test==1]
#     nControls = y_test[y_test==0]


#     X_train_app.append(X_train)
#     X_test_app.append(X_test)
#     y_train_app.append(y_train)
#     y_test_app.append(y_test)

#     y_train, y_test = np.ravel(y_train), np.ravel(y_test)

#     # clf = xgboost.XGBClassifier()
#     clf = LogisticRegression(solver='lbfgs')
#     # tuning 
#     # logreg = LogisticRegression()
#     # param_grid = {'penalty' : ['l1', 'l2'],
#     # 'C' : np.logspace(-4, 4, 20),
#     # 'solver' : ['lbfgs'],
#     # 'random_state': [42]}
    
#     # clf = GridSearchCV(logreg, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)
#     # model = clf.fit(X_val, y_val)
#     # params = model.best_params_
#     # clf = LogisticRegression(**params)
#     model = clf.fit(X_train, y_train)

#     model_p_values = sm.Logit(y_train, X_train).fit()
#     predictions = model.predict(X_test)
#     y_score = model.predict_proba(X_test)[:, 1]
#     precision, recall, thresholds = precision_recall_curve(y_test, y_score)
#     auc_precision_recall = auc(recall, precision)
#     fpr, tpr, threshold = metrics.roc_curve(y_test, y_score)
#     # get the best threshold
#     J = tpr - fpr
#     ix = argmax(J)
#     best_thresh = thresholds[ix]
#     y_pred_odds = (model.predict_proba(X_test)[:,1] >= best_thresh).astype(int)
#     tn, fp, fn, tp = confusion_matrix(y_test,y_pred_odds).ravel()



    
#     odds = (tp*tn)/(fp*fn)
#     if (fp==0 or fn==0):
#         odds=0
#     noSkill = len(y_score[y_test==1]) / len(y_test)
    

#     try:
#         auc_score = roc_auc_score(y_test, y_score)
#     except ValueError:
#         auc_score = 0

#     # Data to plot precision - recall curve
#     precision, recall, thresholds = precision_recall_curve(y_test, y_score)
#     AUPRC_app.append(auc(recall, precision))


#     positive_labels = y_score[y_test==1]
#     negative_labels = y_score[y_test==0]
#     _, p_value = mannwhitneyu(positive_labels, negative_labels)

#     #for logistic reg
#     coeffs = pd.DataFrame(zip(X_train.columns, np.transpose(clf.coef_.tolist()[0])), columns=['features', 'coef'])
#     df_coef.append(coeffs)

        
#     dict_perf['AUPRC'].append(auc(recall, precision))
#     dict_perf['AUROC'].append(auc_score)
#     dict_perf['Odds'].append(odds)
#     dict_perf['noSkill'].append(noSkill)
#     dict_perf['p_value'].append(p_value)
#     dict_perf['nControls'].append(len(nControls))
#     dict_perf['nCases'].append(len(nCases))


#     # Use AUC function to calculate the area under the curve of precision recall curve
#     lab1 = 'Fold %d AUPRC=%.4f' % (i+1, auc_precision_recall)
#     lab2 = 'Fold %d AUROC=%.4f' % (i+1, auc_score)

#     y_real.append(y_test)
#     y_proba.append(y_score)
#     fpr_app.append(fpr)
#     tpr_app.append(tpr)
#     odds_ratio.append(odds)
#     model_app.append(model)

#     axes[1].step(recall, precision, label=lab1)
#     axes[0].step(fpr, tpr, label=lab2)


# write_json_c(path=os.path.join(analysis_folder,"result_logistic_withcoefs_medications+allfeatures_ATC3_deadExl_aggr_14to26"), data = dict_perf)

# y_real = np.concatenate(y_real)
# y_proba = np.concatenate(y_proba)

# fpr, tpr, _ = metrics.roc_curve(y_real,  y_proba)

# precision, recall, _ = precision_recall_curve(y_real, y_proba)

# lab1 = "Overall AUPRC= %.4f, stdev %.4f." % (np.mean(dict_perf['AUPRC']), np.std(dict_perf['AUPRC']))
# lab2 = "Overall AUROC= %.4f, stdev %.4f." % (np.mean(dict_perf['AUROC']), np.std(dict_perf['AUROC']))
# # lab3 = "'Overall Odds= %.4f" % (np.mean(dict_perf['Odds']))

# # axes[0].step(fpr, tpr, label=[lab2,lab3], color='black')
# axes[0].step(fpr, tpr, label=lab2, color='black')
# axes[0].set_xlabel('FPR')
# axes[0].set_ylabel('TPR')
# axes[0].legend(loc='lower right', fontsize='small')


# axes[1].step(recall, precision, label=lab1, color='black')
# axes[1].set_xlabel('Recall')
# axes[1].set_ylabel('Precision')
# axes[1].legend(loc='lower right', fontsize='small')
# no_skill = len(y_proba[y_real==1]) / len(y_real)
# axes[1].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

# fig1.suptitle('Medications', fontsize=12)

# path = os.path.join(analysis_folder)
# fig1.tight_layout()
# fig1.savefig(os.path.join(path+'result_crossval_medications_reducedPredictors_logistic_ATC3_deadexcl_aggr_14to16weeks.pdf'))
# plt.close()


##--- calculate bivariate odds 

variables = X.columns
tmp = pd.DataFrame()

for loc,i in enumerate(df_vars):
    if (i!='GA'): 
        X_train = df_vars[[i, 'GA']]
    # X_ind = df_vars[[i]]
        model = sm.Logit(y,X_train).fit(method='bfgs')
        params = model.params
        conf = model.conf_int()
        conf['coef'] = params
        conf.columns=['5%', '95%', 'coeff']
        conf['Odds'] = np.exp(conf['coeff'] )
        conf['5%'] = np.exp(conf['5%'])
        conf['95%'] = np.exp(conf['95%'])
        conf['log_pvalue'] = -np.log10(model.pvalues)
        conf['odds_pvalue_regression'] = model.pvalues
        y_score = model.predict(X_train, 'prob')
        positive_labels = y_score[y==1]
        negative_labels = y_score[y==0]
        _, p_value = mannwhitneyu(positive_labels, negative_labels)
        conf = conf.drop('GA')



        conf['U_pvalue'] = -np.log10(p_value)
        tmp = pd.concat([tmp,conf], axis=0)


method = 'fdr_bh'
pvals = smt.multipletests (tmp['odds_pvalue_regression'].values, method=method)
tmp['p_value_regression_adj'] = pvals[1]
tmp['include'] = pvals[0]

tmp.to_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/medication_ATC3_bivariate_oddsratio_aggr_1213_14weeksto26_all_notadjusted_GAincluded.csv")

conf= tmp

method='fdr_bh'
conf ['p_value_regression'] = np.power(10,-tmp['log_pvalue'])
pvals = smt.multipletests (tmp['p_value_regression'].values, method=method)
conf['p_value_regression_adj'] = pvals[1]
conf['include'] = pvals[0]
conf = conf[conf['include']]





plt.figure(figsize=(20,10))
ci = [conf.iloc[::-1]['Odds'] - conf.iloc[::-1]['5%'].values, conf.iloc[::-1]['95%'].values - conf.iloc[::-1]['Odds']]
plt.errorbar(x=conf.iloc[::-1]['Odds'], y=conf.iloc[::-1].index.values, xerr=ci,
            color='black',  capsize=3, linestyle='None', linewidth=1,
            marker="o", markersize=5, mfc="black", mec="black")
plt.axvline(x=1, linewidth=0.8, linestyle='--', color='black')
plt.tick_params(axis='both', which='major', labelsize=8)
plt.xlabel('Odds Ratio and 95% Confidence Interval', fontsize=8)
plt.tight_layout()
plt.savefig('/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Figures/medication_allfeatures__raw_forest_plot_ATC3_limitedDrugsBefore14Weeks_FLTRxNorm_deadFLT_aggr_corrected.pdf')