#!/usr/bin/env python
# coding: utf-8

# In[9]:


#loading required packages
get_ipython().run_line_magic('reset', '')
#from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import regex as re


# In[50]:


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'/home/npayrov/.config/gcloud/application_default_credentials.json' 
client = bigquery.Client(project='som-nero-phi-naghaeep-starr')


# In[37]:


# list_concept_ga = [44789950, 4210896, 40485048, 40760825, 40759202, 4260747]
# Query_notes = """

# select * from `som-nero-phi-naghaeep-starr.rit_data_neelufar.note`
# where note_id in (select note_id
#    from `som-nero-phi-naghaeep-starr.rit_data_neelufar.note_nlp`
#    where note_id in (select note_id from `som-nero-phi-naghaeep-starr.rit_data_neelufar.note` where person_id in (select person_id from `som-nero-phi-naghaeep-starr.neelu_explore.BabyMap_MRN_ID`)) and
#       (note_nlp_concept_id IN UNNEST (@list_concept_ga))
#    order by note_id) 
# order by person_id, note_DATE


# """

# job_config = bigquery.QueryJobConfig(
#     query_parameters=[
#         bigquery.ArrayQueryParameter("list_concept_ga", "INT64", list_concept_ga)])


# In[38]:


# baby_notes = client.query(Query_notes, job_config=job_config).to_dataframe()
# baby_notes.to_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/data/baby_notes.csv", index=False)


# In[6]:


baby_notes = None
## read the notes
baby_notes = pd.read_csv("/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/data/baby_notes.csv")
baby_notes['note_text'] = baby_notes['note_text'].apply(lambda x: x.lower())
note_find_ga = pd.DataFrame(baby_notes['note_text'].str.partition("gestational age: ")[2])
note_find_ga_weekDay = pd.DataFrame(note_find_ga[2].str.partition(" ")[0])
note_find_ga_weekDay = note_find_ga_weekDay[0].astype('string')
note_find_ga_weekDay['weekday'] = note_find_ga_weekDay.str.split("w|d", expand=True)
note_find_ga_weekDay['w'] = note_find_ga_weekDay['weekday'][0]
note_find_ga_weekDay['d'] = note_find_ga_weekDay['weekday'][1]
baby_notes = baby_notes.join(note_find_ga_weekDay['w'])
baby_notes = baby_notes.join(note_find_ga_weekDay['d'])
baby_notes.rename(columns={0: 'GA_weeks', 1: 'GA_days'}, inplace=True)
baby_notes['GA_weeks']=pd.to_numeric(baby_notes['GA_weeks'], errors='coerce').astype(float)
baby_notes['GA_days']=pd.to_numeric(baby_notes['GA_days'], errors='coerce').astype(float)
baby_notes = baby_notes[~baby_notes['GA_weeks'].isna()]
baby_notes['GA_days'] = baby_notes['GA_days'].replace(np.nan, 0)
baby_notes['Total_GA_Days'] = (baby_notes['GA_weeks']*7)+baby_notes['GA_days']
baby_ga_id = baby_notes[['person_id', 'Total_GA_Days']]
baby_ga_id_grp = pd.DataFrame(baby_ga_id.groupby('person_id')['Total_GA_Days'].agg('max'))
baby_ga_id_grp['Total_GA_Days'].isnull().sum()


# In[16]:


## number of preterm babies with GA info in notes table
baby_notes[baby_notes['Total_GA_Days']<37*7].person_id.nunique()


# In[15]:


## number of all babies with GA info in notes table

baby_notes.person_id.nunique()


# In[4]:


## term and preterm babies (person_ids) the threshold is 37 weeks (37*7=259 days)
thresh = (37*7)
term_baby_ids = pd.DataFrame(baby_notes[baby_notes['Total_GA_Days']>=thresh].person_id.unique())


# In[10]:


# term_baby_ids.to_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/data/term_baby_ids.csv", index=False)
term_baby_ids = pd.read_csv("/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/data/term_baby_ids.csv")


# In[11]:


## separate readings for measurements of interest for term and pre term
# baby_measures = pd.read_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/analysis/pivot_chuncked_all195.csv",\
#                                             index_col=False).drop(columns=['Unnamed: 0'], errors='ignore')


# In[12]:


baby_measures = pd.read_csv("/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv")




# In[13]:


baby_measures.shape


# In[14]:


meas_of_int = ['Neutrophils/100 leukocytes in Blood', 'Monocytes/100 leukocytes in Blood'\
              ,"Lymphocytes/100 leukocytes in Blood", "Basophils/100 leukocytes in Blood", \
               "Eosinophils/100 leukocytes in Blood"]
baby_measures_int = baby_measures[['person_id', meas_of_int[0], meas_of_int[1],meas_of_int[2],meas_of_int[3],\
                                   meas_of_int[4],'age_at_meas']]


# In[15]:


baby_measures_int_GA = baby_measures_int.join(baby_ga_id_grp,on='person_id')
baby_measures_int_GA['age_meas_GA'] = (baby_measures_int_GA['age_at_meas']*7)+baby_measures_int_GA['Total_GA_Days']
baby_measures_int_GA['age_at_meas_round'] = baby_measures_int_GA['age_at_meas'].round(0).astype(int)
baby_measures_int_GA['age_at_meas_str'] = baby_measures_int_GA['age_at_meas'].astype(str)


# In[57]:


baby_measures_int_GA['age_at_meas_str'] = baby_measures_int_GA['age_at_meas'].astype(str)


# In[58]:


df_quant_neut = baby_measures_int_GA[['person_id', meas_of_int[0],'age_at_meas','age_meas_GA','Total_GA_Days','age_at_meas_round','age_at_meas_str']]
df_quant_mono = baby_measures_int_GA[['person_id', meas_of_int[1],'age_at_meas','age_meas_GA','Total_GA_Days','age_at_meas_round','age_at_meas_str']]
df_quant_lymph = baby_measures_int_GA[['person_id', meas_of_int[2],'age_at_meas','age_meas_GA','Total_GA_Days','age_at_meas_round','age_at_meas_str']]
df_quant_baso = baby_measures_int_GA[['person_id', meas_of_int[3],'age_at_meas','age_meas_GA','Total_GA_Days','age_at_meas_round','age_at_meas_str']]
df_quant_eosi = baby_measures_int_GA[['person_id', meas_of_int[4],'age_at_meas','age_meas_GA','Total_GA_Days','age_at_meas_round','age_at_meas_str']]




# In[59]:


df_quant_neut ['Term'] = np.where(df_quant_neut['Total_GA_Days']>=259, 1, 0)
df_quant_mono ['Term'] = np.where(df_quant_mono['Total_GA_Days']>=259, 1, 0)
df_quant_lymph['Term'] = np.where(df_quant_lymph['Total_GA_Days']>=259, 1, 0)
df_quant_baso['Term'] = np.where(df_quant_baso['Total_GA_Days']>=259, 1, 0)
df_quant_eosi['Term'] = np.where(df_quant_eosi['Total_GA_Days']>=259, 1, 0)


# In[60]:


##--- plotting all together
fig7, axs = plt.subplots(5,1,figsize=(20,30), sharex=True)
axs = axs.ravel()
sns.set_theme(style='white')
medianprops = dict(color="black",linewidth=1.5)
my_colors = ["salmon", 'cornflowerblue']
sns.set_palette( my_colors )



l0 = sns.lineplot(x=df_quant_neut['age_at_meas'], y=df_quant_neut['Neutrophils/100 leukocytes in Blood'], hue \
                      =df_quant_neut['Term'], ax=axs[0])
l1 = sns.lineplot(x=df_quant_baso['age_at_meas'], y=df_quant_baso["Basophils/100 leukocytes in Blood"], hue \
                      =df_quant_baso['Term'], ax=axs[1])
l2 = sns.lineplot(x=df_quant_lymph['age_at_meas'], y=df_quant_lymph["Lymphocytes/100 leukocytes in Blood"], hue \
                      =df_quant_lymph['Term'], ax=axs[2])
l3 = sns.lineplot(x=df_quant_eosi['age_at_meas'], y=df_quant_eosi["Eosinophils/100 leukocytes in Blood"], hue \
                      =df_quant_eosi['Term'], ax=axs[3])
l4 = sns.lineplot(x=df_quant_mono['age_at_meas'], y=df_quant_mono["Monocytes/100 leukocytes in Blood"], hue \
                      =df_quant_mono['Term'], ax=axs[4])


legend = axs[0].legend(fontsize = 20)
legend.texts[0].set_text("")
axs[0].get_legend().get_texts()[0].set_text('pre-term')
axs[0].get_legend().get_texts()[1].set_text('term')


axs[0].set_ylabel("Neutrophils",fontsize=40)
axs[1].set_ylabel("Basophils",fontsize=40)
axs[2].set_ylabel("Lymphocytes",fontsize=40)
axs[3].set_ylabel("Eosinophils",fontsize=40)
axs[4].set_ylabel("Monocytes",fontsize=40)

axs[0].set_xlabel("")
axs[1].set_xlabel("")
axs[2].set_xlabel("")
axs[3].set_xlabel("")
axs[4].set_xlabel("Age at measurement (weeks)",fontsize=40)

axs[0].tick_params(axis='both', labelsize=16)
axs[1].tick_params(axis='both', labelsize=16)
axs[2].tick_params(axis='both', labelsize=16)
axs[3].tick_params(axis='both', labelsize=16)
axs[4].tick_params(axis='both', labelsize=16)

axs[0].xaxis.set_tick_params(labelbottom=True)
axs[1].xaxis.set_tick_params(labelbottom=True)
axs[2].xaxis.set_tick_params(labelbottom=True)
axs[3].xaxis.set_tick_params(labelbottom=True)
axs[4].xaxis.set_tick_params(labelbottom=True)

axs[1].get_legend().remove()
axs[2].get_legend().remove()
axs[3].get_legend().remove()
axs[4].get_legend().remove()




fig7.tight_layout()



# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns

#--- plotting all together
fig7, axs = plt.subplots(5, 1, figsize=(20, 30), sharex=True)
axs = axs.ravel()
sns.set_theme(style='white')

# Line thickness and colors
medianprops = dict(color="black", linewidth=1.5)
my_colors = ["salmon", 'cornflowerblue']
sns.set_palette(my_colors)

# Plot with no shading and thicker lines
l0 = sns.lineplot(x=df_quant_neut['age_at_meas'], 
                  y=df_quant_neut['Neutrophils/100 leukocytes in Blood'], 
                  hue=df_quant_neut['Term'], ax=axs[0], 
                  errorbar=None, linewidth=3)

l1 = sns.lineplot(x=df_quant_baso['age_at_meas'], 
                  y=df_quant_baso["Basophils/100 leukocytes in Blood"], 
                  hue=df_quant_baso['Term'], ax=axs[1], 
                  errorbar=None, linewidth=3)

l2 = sns.lineplot(x=df_quant_lymph['age_at_meas'], 
                  y=df_quant_lymph["Lymphocytes/100 leukocytes in Blood"], 
                  hue=df_quant_lymph['Term'], ax=axs[2], 
                  errorbar=None, linewidth=3)

l3 = sns.lineplot(x=df_quant_eosi['age_at_meas'], 
                  y=df_quant_eosi["Eosinophils/100 leukocytes in Blood"], 
                  hue=df_quant_eosi['Term'], ax=axs[3], 
                  errorbar=None, linewidth=3)

l4 = sns.lineplot(x=df_quant_mono['age_at_meas'], 
                  y=df_quant_mono["Monocytes/100 leukocytes in Blood"], 
                  hue=df_quant_mono['Term'], ax=axs[4], 
                  errorbar=None, linewidth=3)

# Customize legends
legend = axs[0].legend(fontsize=20)
legend.texts[0].set_text("")
axs[0].get_legend().get_texts()[0].set_text('pre-term')
axs[0].get_legend().get_texts()[1].set_text('term')

# Set labels
axs[0].set_ylabel("Neutrophils", fontsize=40)
axs[1].set_ylabel("Basophils", fontsize=40)
axs[2].set_ylabel("Lymphocytes", fontsize=40)
axs[3].set_ylabel("Eosinophils", fontsize=40)
axs[4].set_ylabel("Monocytes", fontsize=40)

axs[4].set_xlabel("Age at measurement (weeks)", fontsize=40)

# Remove x-labels for top plots
for i in range(4):
    axs[i].set_xlabel("")

# Ticks size
for ax in axs:
    ax.tick_params(axis='both', labelsize=16)
    ax.xaxis.set_tick_params(labelbottom=True)

# Remove extra legends
for i in range(1, 5):
    axs[i].get_legend().remove()

fig7.tight_layout()
plt.show()


# In[25]:


fig7.tight_layout()
fig7.savefig("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Figures/allWBC_trend_GA.png", dpi=300)
fig7.savefig("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Figures/allWBC_trend_GA.pdf")


