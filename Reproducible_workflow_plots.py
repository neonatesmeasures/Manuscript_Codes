#!/usr/bin/env python
# coding: utf-8

# In[1]:


#loading required packages
get_ipython().run_line_magic('reset', '')
from google.cloud import bigquery
import os
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import nanmean


# In[2]:


## read in the dataset which includes the measurements and their units

df_measures_units = pd.read_csv("/Users/sshome/Desktop/Bioinformatics_projects/Neelu_project/Neonatal_measurements_final_share/Data/babies_measurements_units.csv")




# In[3]:


# df_measures_units['measurement_name'].to_list()


# In[4]:


## function to list all the features along with the percentage of the majority unit they have been recorded under
## we use this to decide which measurements to exclude
## criteria: 1. if a measure is recorded under various units and the majority unit is just around 40-50 times, we manually check and
## if we see so many units we exlude the entire meaurement
## 2. For the rest of the measurements, we exlcude the records under minority units 
## this should be all done before removing outliers 

def units_per_measure_percent (df):
    dict = {} 
    for i,j in enumerate(df['measurement_name'].unique()):
        print(i,j)
        df_tmp = df[df['measurement_name']==j]['unit_name'].value_counts().reset_index()
        if df_tmp.empty:
            dict[i] = {'variable':j, "maj_unit":np.nan , 'percent':np.nan}
        else:
            sum_ = df[df['measurement_name']==j]['unit_name'].value_counts().reset_index().sum()[1]
            max_count = df_tmp[df_tmp['unit_name']==df_tmp['unit_name'].max()]['unit_name'][0]
            max_unit = df_tmp[df_tmp['unit_name']==df_tmp['unit_name'].max()]['index'][0]

            percent = (np.float64(max_count)/np.float64(sum_))*100

            dict[i] = {'variable':j, "maj_unit":max_unit, 'percent':percent}
            df_units_measures = pd.DataFrame(dict).T
            df_units_measures['percent'] = df_units_measures['percent'].astype(float)
    return df_units_measures 





# In[5]:


df_units_measures = units_per_measure_percent(df_measures_units)


# In[6]:


df_units_measures


# In[19]:


filtered_df = df_units_measures[df_units_measures['variable'].str.contains('lymphocytes', case=False)]


# In[20]:


filtered_df


# In[22]:


filtered_df.to_csv('lymphocytes_concept_list.csv', index=False)


# In[23]:


pwd


# In[17]:


## save the percentages of the units that have the majority of the recordings for future use 

df_units_measures = df_units_measures.sort_values(by='percent')
# df_units_measures.to_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/babies_measures_units_counted_all_measurement_new_RIT_update.csv", index=False)


# In[ ]:





# In[18]:


## number of all features 
df_units_measures['variable'].nunique()


# In[19]:


## we exclude all the measurements that are not recorded under 100% unique units except for heart rate and Body weight
list_of_variables_to_exclude = df_units_measures[(df_units_measures.percent <100) & ~(df_units_measures.variable.isin(['Heart rate', 'Body weight']))].variable





# In[20]:


list_of_variables_to_exclude


# In[21]:


list_of_variables_to_exclude.to_csv('test.csv', index=False)


# In[22]:


## importing the measurements data in from GCP 

#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'/home/npayrov/.config/gcloud/application_default_credentials.json' 

#client = bigquery.Client(project='som-nero-phi-naghaeep-starr')


Query_measures = """
SELECT DISTINCT * from 
(SELECT m.person_id, p.birth_DATETIME, c.concept_name as measurement_name, m.measurement_concept_id,m.measurement_DATE, m.measurement_DATETIME,m.value_as_number,m.unit_concept_id,c2.concept_name as unit_name 

FROM `som-nero-phi-naghaeep-starr.rit_data_neelufar.measurement_20221011` as m
left join `som-nero-phi-naghaeep-starr.rit_data_neelufar.person` as p
on m.person_id = p.person_id
left join `som-nero-phi-naghaeep-starr.rit_data_neelufar.concept_20221011` as c
on m.measurement_concept_id = c.concept_id
left join `som-nero-phi-naghaeep-starr.rit_data_neelufar.concept_20221011` as c2
on m.unit_concept_id = c2.concept_id

WHERE m.person_id IN (select person_id from `som-nero-phi-naghaeep-starr.neelu_explore.BabyMap_MRN_ID`)
AND m.person_id NOT IN (select person_id from `som-nero-phi-naghaeep-starr.neelu_explore.Baby_death_info`)
AND m.value_as_number IS NOT NULL
AND DATE_DIFF (m.measurement_DATETIME,p.birth_DATETIME, WEEK)>=0 and DATE_DIFF (m.measurement_DATETIME,p.birth_DATETIME, WEEK)<=26
)
"""

df_measures = client.query(Query_measures).to_dataframe()

# remove not matching concepts
df_measures = df_measures[df_measures.measurement_name != 'No matching concept']

# df_measures_outlier_rmv = df_measures[df_measures.groupby("concept_name").value_as_number.\
#       transform(lambda x : (x>(x.quantile(0.25)) - 1.5*(x.quantile(0.75)-x.quantile(0.25)))&(x<x.quantile(0.75)+1.5*(x.quantile(0.75)-x.quantile(0.25))).eq(1))]

# df_measures_outlier_rmv_agg = df_measures_outlier_rmv.groupby(['person_id','birth_DATETIME','concept_name','measurement_DATE'])['value_as_number'].agg(np.mean).reset_index()
# df_measures_outlier_rmv_pvt  = df_measures_outlier_rmv_agg.pivot_table(values='value_as_number', index=['person_id', 'birth_DATETIME',
#                                                                       'measurement_DATE'], \
#                                                    columns='concept_name').reset_index().drop_duplicates()
# df_measures_outlier_rmv_pvt['age_at_meas'] = ((pd.to_datetime(df_measures_outlier_rmv_pvt['measurement_DATE'])-pd.to_datetime(df_measures_outlier_rmv_pvt['birth_DATETIME'])).dt.days)/7
# df_measures_outlier_rmv_pvt_6months = df_measures_outlier_rmv_pvt[(df_measures_outlier_rmv_pvt['age_at_meas']>=0) &(df_measures_outlier_rmv_pvt['age_at_meas']<=26) ]




# In[23]:


## exclude unwanted variables 
df_measures =  pd.DataFrame(df_measures[~df_measures['measurement_name'].isin(list_of_variables_to_exclude)])


# In[ ]:


df_measures['measurement_name'].nunique()


# In[ ]:


## converting all body wieght values to pounds 
def convert_all_weight_to_pounds(df):
    for i,j in enumerate (df['measurement_name']):
        if ((j == 'Body weight') and (df['unit_name'].get(i,0) == 'ounce (avoirdupois)')):
            df['value_as_number'][i] = (np.float64(df['value_as_number'][i]))/16
        elif ((j == 'Body weight') and (df['unit_name'].get(i,0)=='kilogram')):
            df['value_as_number'][i] = (np.float64(df['value_as_number'][i]))*2.205
    return df



# In[ ]:


df_measures_converted = convert_all_weight_to_pounds(df_measures)


# In[ ]:


## remove outliers: for each measurement, values beyond 3 stdev (lower and upper) are removed
df_measures_outlier_rmv = df_measures_converted[df_measures_converted.value_as_number.\
          transform(lambda x : (x>np.mean(x)-np.std(x)*3)&(x<np.mean(x)+np.std(x)*3).eq(1))]


# In[ ]:


df_measures_outlier_rmv.shape


# In[ ]:


neut_auto = 'Neutrophils/100 leukocytes in Blood by Automated count'
neut_man = 'Neutrophils/100 leukocytes in Blood by Manual count'
baso_auto = 'Basophils/100 leukocytes in Blood by Automated count'
baso_man = 'Basophils/100 leukocytes in Blood by Manual count'
lymph_auto = 'Lymphocytes/100 leukocytes in Blood by Automated count'
lymph_man = 'Lymphocytes/100 leukocytes in Blood by Manual count'
mono_auto = 'Monocytes/100 leukocytes in Blood by Automated count'
mono_man = 'Monocytes/100 leukocytes in Blood by Manual count'
eosi_auto = 'Eosinophils/100 leukocytes in Blood by Automated count'
eosi_man = 'Eosinophils/100 leukocytes in Blood by Manual count'


df_measures_outlier_rmv.loc[df_measures_outlier_rmv["measurement_name"] == neut_auto, "measurement_name"] = 'Neutrophils/100 leukocytes in Blood'
df_measures_outlier_rmv.loc[df_measures_outlier_rmv["measurement_name"] == neut_man, "measurement_name"] = 'Neutrophils/100 leukocytes in Blood'
df_measures_outlier_rmv.loc[df_measures_outlier_rmv["measurement_name"] == baso_auto, "measurement_name"] = 'Basophils/100 leukocytes in Blood'
df_measures_outlier_rmv.loc[df_measures_outlier_rmv["measurement_name"] == baso_man, "measurement_name"] = 'Basophils/100 leukocytes in Blood'
df_measures_outlier_rmv.loc[df_measures_outlier_rmv["measurement_name"] == lymph_auto, "measurement_name"] = 'Lymphocytes/100 leukocytes in Blood'
df_measures_outlier_rmv.loc[df_measures_outlier_rmv["measurement_name"] == lymph_man, "measurement_name"] = 'Lymphocytes/100 leukocytes in Blood'
df_measures_outlier_rmv.loc[df_measures_outlier_rmv["measurement_name"] == mono_auto, "measurement_name"] = 'Monocytes/100 leukocytes in Blood'
df_measures_outlier_rmv.loc[df_measures_outlier_rmv["measurement_name"] == mono_man, "measurement_name"] = 'Monocytes/100 leukocytes in Blood'
df_measures_outlier_rmv.loc[df_measures_outlier_rmv["measurement_name"] == eosi_auto, "measurement_name"] = 'Eosinophils/100 leukocytes in Blood'
df_measures_outlier_rmv.loc[df_measures_outlier_rmv["measurement_name"] == eosi_man, "measurement_name"] = 'Eosinophils/100 leukocytes in Blood'


# In[ ]:


## saving units of measures
# df_units = df_measures_outlier_rmv[['measurement_name','unit_name']].drop_duplicates()
# df_units.groupby('measurement_name')['unit_name'].first().reset_index()

# df_units['measurement_name'] = df_units['measurement_name'].apply(lambda x: ''.join(filter(str.isalnum, str(x))))
# df_units.to_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/measure_units.csv", index=False)


# In[85]:


df_measures_outlier_rmv_agg = df_measures_outlier_rmv.groupby(['person_id','birth_DATETIME','measurement_name','measurement_DATE'])['value_as_number'].agg(np.mean).reset_index()
df_measures_outlier_rmv_pvt  = df_measures_outlier_rmv_agg.pivot_table(values='value_as_number', index=['person_id', 'birth_DATETIME',
                                                                      'measurement_DATE'], \
                                                   columns='measurement_name').reset_index().drop_duplicates()
df_measures_outlier_rmv_pvt['age_at_meas'] = ((pd.to_datetime(df_measures_outlier_rmv_pvt['measurement_DATE'])-pd.to_datetime(df_measures_outlier_rmv_pvt['birth_DATETIME'])).dt.days)/7
df_measures_outlier_rmv_pvt_6months = df_measures_outlier_rmv_pvt[(df_measures_outlier_rmv_pvt['age_at_meas']>=0) &(df_measures_outlier_rmv_pvt['age_at_meas']<=26) ]


# In[11]:


df_measures_outlier_rmv_pvt_6months


# In[90]:


## testing some WBC patterns

test = df_measures_outlier_rmv_pvt_6months[['Neutrophils/100 leukocytes in Blood', 'age_at_meas']]
sns.lineplot(data= test, x = 'age_at_meas', y = 'Neutrophils/100 leukocytes in Blood' )


# In[ ]:


## testing some WBC patterns

test = df_measures_outlier_rmv_pvt_6months[['Neutrophils/100 leukocytes in Blood', 'age_at_meas']]
sns.lineplot(data= test, x = 'age_at_meas', y = 'Neutrophils/100 leukocytes in Blood' )


# In[91]:


## saving
df_measures_outlier_rmv_pvt_6months.to_parquet("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.gzip", compression='gzip',index=False)




# In[92]:


df_measures_outlier_rmv_pvt_6months.to_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv",index=False)


# In[22]:


df_measures_outlier_rmv_pvt_6months


# In[ ]:





# In[ ]:


## plotting the trends of interes. The colors represent the clusters they belong to in the correlation network
cluster_count = 5
# plt.set_cmap('Set3')
# plt.set_cmap('tab20')
colors = ["#6495ED", "#DA70D6", "#20B2AA", "#FF8C00", "red"]


# km = TimeSeriesKMeans(n_clusters=cluster_count, metric="euclidean", random_state=49)

# labels = km.fit_predict(mySeries)
with open("/Users/sshome/Desktop/Bioinformatics_projects/Neelu_project/Neonatal_measurements_final_share/Data/clustering_labels", "rb") as fp1:   #Unpickling
    labels = pickle.load(fp1)
with open("/Users/sshome/Desktop/Bioinformatics_projects/Neelu_project/Neonatal_measurements_final_share/Data/clustering_measures", "rb") as fp2:   #Unpickling
    mySeries = pickle.load(fp2)
with open("/Users/sshome/Desktop/Bioinformatics_projects/Neelu_project/Neonatal_measurements_final_share/Data/clustering_measures_names", "rb") as fp3:   #Unpickling
    nameofSeries = pickle.load(fp3)




fig, axs = plt.subplots(2,3,layout='constrained',figsize=(30,15))
# palette = itertools.cycle(sns.color_palette('Set3'))
# palette = itertools.cycle(sns.color_palette('Set2'))

axs = axs.ravel()

# For each label there is,
# plots every series with that label

for label,color in zip(set(labels),colors):
    cluster = []
    var_names = []
    vars_of_int = ['Basophils100leukocytesinBlood',\
                   'Neutrophils100leukocytesinBlood',\
                    'Eosinophils100leukocytesinBlood',\
                    'Lymphocytes100leukocytesinBlood',\
                   'Monocytes100leukocytesinBlood'\
                                                        ]
    for i in range(len(labels)):
            if(labels[i]==label and nameofSeries[i] in (vars_of_int)):
#                 axs[label].plot(mySeries[i],c="gray",alpha=0.05)
#                 sns.lineplot(x =mySeries[i].index,y=mySeries[i].values, ax = axs[row_i, column_j])


#         sns.lineplot(x = 'age_at_meas', y = df_scaled_avg.values, data=df_scaled_avg, ax= axs[label],\
#                                 color=color
# #                                 color=next(palette)\)
                var_of_int = nameofSeries[i]
                sns.set_theme(style='white')
#                                
                get_plot = sns.lineplot(x = mySeries[i].index , y=mySeries[i].values, ax= axs[vars_of_int.index(var_of_int)],\
                                color=color
#                                 color=next(palette)\
                               )
#                 axs[vars_of_int.index(var_of_int)].set(ylim=(-4,4))
# #         axs[label].plot(np.average(np.vstack(cluster),axis=0),c="red")
#         axs[label].plot(df_scaled_avg, c=color)
# #         axs[label].set_xlabel('age at measurement (weeks)')
#                 axs[label].set(ylim=(-4,3))



axs[0].set_ylabel('')
axs[1].set_ylabel('')
axs[2].set_ylabel('')
axs[3].set_ylabel('')
axs[4].set_ylabel('')

axs[0].set_xlabel('')
axs[1].set_xlabel('')
axs[2].set_xlabel('')
axs[3].set_xlabel('')
axs[4].set_xlabel('')


axs[0].set(ylim=(-4,4))
axs[1].set(ylim=(-4,4))
axs[2].set(ylim=(-4,4))
axs[3].set(ylim=(-4,4))
axs[4].set(ylim=(-4,4))


axs[0].tick_params(axis='both', which='major', labelsize=20)
axs[1].tick_params(axis='both', which='major', labelsize=20)
axs[2].tick_params(axis='both', which='major', labelsize=20)
axs[3].tick_params(axis='both', which='major', labelsize=20)
axs[4].tick_params(axis='both', which='major', labelsize=20)

axs[0].set_title('Basophils/100 leukocytes in Blood',fontsize=32,pad=20)
axs[1].set_title('Neutrophils/100 leukocytes in Blood',fontsize=32,pad=20) 
axs[2].set_title('Eosinophils/100 leukocytes in Blood',fontsize=32,pad=20)
axs[3].set_title('Lymphocytes/100 leukocytes in Blood',fontsize=32,pad=20)
axs[4].set_title('Monocytes/100 leukocytes in Blood',fontsize=32,pad=20)


fig.supylabel('Value of the measurement (scaled)', fontsize=32)
fig.supxlabel('Age at measurement (weeks)', fontsize=32)


# fig.tight_layout()
fig.delaxes(axs[5])

