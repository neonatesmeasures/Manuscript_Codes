#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reset', '')
from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import regex as re
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import math
from statsmodels.sandbox.stats.multicomp import multipletests
import pickle
import itertools

seed = 49


np.random.seed(seed)

df_measurements_6mnt_outl_rmv = pd.read_csv("/Users/sshome/Desktop/Bioinformatics_projects/Neelu_project/Neonatal_measurements_final_share/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv")
rmv_vars = ["person_id","age_at_meas","birth_DATETIME","measurement_DATE"]      
df = df_measurements_6mnt_outl_rmv.drop(axis=1, labels=rmv_vars, errors='ignore')


# In[2]:


df = df_measurements_6mnt_outl_rmv.drop(axis=1, labels=rmv_vars, errors='ignore')
df.columns = df.apply(lambda x: ''.join(filter(str.isalnum, str(x.name))))
p_values = pd.read_csv("/Users/sshome/Desktop/Bioinformatics_projects/Neelu_project/Neonatal_measurements_final_share/Data/linear_mixed_eff_trends_WBCagg_measuresScaled.csv")
p_values = p_values[~(p_values['p_value']=='error')]
p_values['p_value'] = pd.to_numeric(p_values['p_value'])
p_values['log_10_p_value'] = -np.log10(p_values['p_value'])
max_value = np.nanmax(p_values['log_10_p_value'][p_values['log_10_p_value'] != np.inf])

#replace inf and -inf in column with max value of column 
p_values['log_10_p_value'].replace([np.inf, -np.inf], max_value, inplace=True)
p_values.Variables = p_values.Variables.str.replace('[#,@,&,.,/]', '')
p_values['Variables'].replace('X25hydroxyvitaminD3MassvolumeinSerumorPlasma', '25hydroxyvitaminD3MassvolumeinSerumorPlasma',inplace=True)
p_values['Variables'].replace('X11DeoxycortisolMassvolumeinSerumorPlasma', '11DeoxycortisolMassvolumeinSerumorPlasma',inplace=True)
p_values['Variables'].replace('X17Hydroxyprogesteronemeasurement', '17Hydroxyprogesteronemeasurement',inplace=True)
p_values['Variables'].replace('X2MethylbutyrylglycineCreatinineMassRatioinUrine', '2MethylbutyrylglycineCreatinineMassRatioinUrine',inplace=True)
p_values['Variables'].replace('X25HydroxyvitaminD325HydroxyvitaminD2MassvolumeinSerumorPlasma', '25HydroxyvitaminD325HydroxyvitaminD2MassvolumeinSerumorPlasma',inplace=True)
p_values['Variables'].replace('X25hydroxyvitaminD3MassvolumeinSerumorPlasma', '25hydroxyvitaminD3MassvolumeinSerumorPlasma',inplace=True)
p_values['Variables'].replace('X13betaglucanMassvolumeinSerum','13betaglucanMassvolumeinSerum',inplace=True)
p_values['Variables'].replace('X10HydroxycarbazepineMassvolumeinSerumorPlasma','10HydroxycarbazepineMassvolumeinSerumorPlasma',inplace=True)
p_values['Variables'].replace('X17HydroxyprogesteroneMassvolumeinSerumorPlasma','17HydroxyprogesteroneMassvolumeinSerumorPlasma',inplace=True)

p_values = p_values[p_values['Variables'].isin(df.columns)]


method = 'fdr_bh' #fdr_bh, bonferroni

p_values['adj_p_values'], p_values['Include']= multipletests(p_values['p_value'].to_list(), method=method, returnsorted=False)[1],multipletests(p_values['p_value'].to_list(), method=method, returnsorted=False)[0]
p_values['adj_log_10'] = -np.log10(p_values['adj_p_values'])
max_value = np.nanmax(p_values['adj_log_10'][p_values['adj_log_10'] != np.inf])

#replace inf and -inf in column with max value of column 
p_values['adj_log_10'].replace([np.inf, -np.inf], max_value, inplace=True)
p_values.to_csv("/Users/sshome/Desktop/Bioinformatics_projects/Neelu_project/Neonatal_measurements_final_share/Data/trends_significance_BHadjusted.csv",index=False)
p_values = p_values[p_values['Include']]


df = df.drop(columns=[col for col in df if col not in p_values['Variables'].to_list()])
df = pd.merge(df,df_measurements_6mnt_outl_rmv['age_at_meas'], how='left', left_index=True, right_index=True).sort_values(by='age_at_meas', ascending=True)
df = df.groupby('age_at_meas').agg('mean').reset_index()


df.set_index('age_at_meas',inplace=True)
df.sort_index(inplace=True)

mySeries = []
nameofSeries = []

my_pd = pd.DataFrame()

for i,j in enumerate(df.columns):
    ser = df.iloc[:,i]
    ser = ser.interpolate()
    ser = ser.ffill().bfill()
    scaler = StandardScaler()
    ser_scaled = scaler.fit_transform(np.array(ser).reshape(-1,1))
    ser_scaled_indx = pd.Series(ser_scaled.ravel()).set_axis(ser.index)


    mySeries.append(ser_scaled_indx)
    nameofSeries.append(ser.name)

# with open("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/clustering_measures", "wb") as fp:   #Pickling
#     pickle.dump(mySeries, fp)
# with open("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/clustering_measures_names", "wb") as fp:   #Pickling
#     pickle.dump(nameofSeries, fp)



# In[3]:


distortions = []
for i in range(1, 16):
    km = TimeSeriesKMeans(n_clusters=i, metric="euclidean",init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=49)
#     km = KMeans(n_clusters=i)
    km.fit(mySeries)
    distortions.append(km.inertia_)

#plot 
plt.plot(range(1, 16), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


# In[4]:


cluster_count = 5


km = TimeSeriesKMeans(n_clusters=cluster_count, metric="euclidean")
km = KMeans(n_clusters=cluster_count, random_state=49)
labels = km.fit_predict(mySeries)
som_x = som_y = math.ceil(math.sqrt(cluster_count))

with open("/Users/sshome/Desktop/Bioinformatics_projects/Neelu_project/Neonatal_measurements_final_share/Data/clustering_labels", "wb") as fp:   #Pickling
    pickle.dump(labels, fp)

plot_count = math.ceil(math.sqrt(cluster_count))
fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))
fig.suptitle('Clusters')
row_i=0
column_j=0
# For each label there is,
# plots every series with that label
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
            if(labels[i]==label):

                cluster.append(mySeries[i])
    if len(cluster) > 0:
        sns.set()
        regular_list = cluster
        flat_list = [item for sublist in regular_list for item in sublist]
        regular_list_x = [cluster[i].index.values for i in range(len(cluster))]
        flat_list_x = [item for sublist in regular_list_x for item in sublist]
        data = {'y': flat_list, 'x': flat_list_x}
        df_plot = pd.DataFrame(data)
        get_plot = sns.lineplot(x = 'x'\
                                ,y = 'y',data=df_plot, ax= axs[row_i, column_j])
        get_plot.set(ylim=(-5,5))
    axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
    column_j+=1
    if column_j%plot_count == 0:
        row_i+=1
        column_j=0



# In[5]:


cluster_count = 5

colors = ["#6495ED", "#DA70D6", "#20B2AA", "#FF8C00","red"]


# km = TimeSeriesKMeans(n_clusters=cluster_count, metric="euclidean", random_state=49)

# labels = km.fit_predict(mySeries)
# som_x = som_y = math.ceil(math.sqrt(cluster_count))

with open("/Users/sshome/Desktop/Bioinformatics_projects/Neelu_project/Neonatal_measurements_final_share/Data/clustering_labels", "rb") as fp1:   #Unpickling
    labels = pickle.load(fp1)
with open("/Users/sshome/Desktop/Bioinformatics_projects/Neelu_project/Neonatal_measurements_final_share/Data/clustering_measures", "rb") as fp2:   #Unpickling
    mySeries = pickle.load(fp2)
with open("/Users/sshome/Desktop/Bioinformatics_projects/Neelu_project/Neonatal_measurements_final_share/Data/clustering_measures_names", "rb") as fp3:   #Unpickling
    nameofSeries = pickle.load(fp3)


# plot_count = math.ceil(math.sqrt(cluster_count))

fig, axs = plt.subplots(2,3,figsize=(15,10))
# palette = itertools.cycle(sns.color_palette('Set3'))
# palette = itertools.cycle(sns.color_palette('Set2'))

axs = axs.ravel()

# For each label there is,
# plots every series with that label

for label,color in zip(set(labels),colors):
    cluster = []
    var_names = []
    for i in range(len(labels)):
            if(labels[i]==label):
#                 axs[label].plot(mySeries[i],c="gray",alpha=0.05)
#                 sns.lineplot(x =mySeries[i].index,y=mySeries[i].values, ax = axs[row_i, column_j])
                cluster.append(mySeries[i])
                var_names.append(mySeries[i].name)


    if len(cluster) > 0:

        regular_list = cluster
        flat_list = [item for sublist in regular_list for item in sublist]
        regular_list_x = [cluster[i].index.values for i in range(len(cluster))]
        flat_list_x = [item for sublist in regular_list_x for item in sublist]
        data = {'y': flat_list, 'x': flat_list_x}
        df_plot = pd.DataFrame(data)

#         sns.lineplot(x = 'age_at_meas', y = df_scaled_avg.values, data=df_scaled_avg, ax= axs[label],\
#                                 color=color
# #                                 color=next(palette)\)
        sns.set_theme(style='white')
#                                
        get_plot = sns.lineplot(x = 'x', y='y', data=df_plot, ax= axs[label],\
                                color=color
#                                 color=next(palette)\
                               )
        axs[label].set(ylim=(-5,5))
# #         axs[label].plot(np.average(np.vstack(cluster),axis=0),c="red")
#         axs[label].plot(df_scaled_avg, c=color)
        axs[label].set_title("Cluster "+str(label))
# #         axs[label].set_xlabel('age at measurement (weeks)')

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

fig.supylabel('Value of the measurement (scaled)')
fig.supxlabel('Age at measurement (weeks)')
fig.tight_layout()


fig.delaxes(axs[5])

# #     column_j+=1
# #     if column_j%plot_count == 0:
# #         row_i+=1
# #         column_j=0




# In[6]:


cluster_count = 5
# plt.set_cmap('Set3')
# plt.set_cmap('tab20')
colors = ["#6495ED", "#DA70D6", "#20B2AA", "#FF8C00","red"]

with open("/Users/sshome/Desktop/Bioinformatics_projects/Neelu_project/Neonatal_measurements_final_share/Data/clustering_labels", "rb") as fp1:   #Unpickling
    labels = pickle.load(fp1)
with open("/Users/sshome/Desktop/Bioinformatics_projects/Neelu_project/Neonatal_measurements_final_share/Data/clustering_measures", "rb") as fp2:   #Unpickling
    mySeries = pickle.load(fp2)
with open("/Users/sshome/Desktop/Bioinformatics_projects/Neelu_project/Neonatal_measurements_final_share/Data/clustering_measures_names", "rb") as fp3:   #Unpickling
    nameofSeries = pickle.load(fp3)


## Plot separately 

fig1,axs1 = plt.subplots(1,1,figsize=(20,10))
fig2,axs2 = plt.subplots(1,1,figsize=(20,10))
fig3,axs3 = plt.subplots(1,1,figsize=(20,10))
fig4,axs4 = plt.subplots(1,1,figsize=(20,10))
fig5,axs5 = plt.subplots(1,1,figsize=(20,10))

for label,color in zip(set(labels),colors):
    cluster = []
    var_names = []
    for i in range(len(labels)):
            if(labels[i]==label):
                cluster.append(mySeries[i])
                var_names.append(mySeries[i].name)


    if len(cluster) > 0:

        regular_list = cluster
        flat_list = [item for sublist in regular_list for item in sublist]
        regular_list_x = [cluster[i].index.values for i in range(len(cluster))]
        flat_list_x = [item for sublist in regular_list_x for item in sublist]
        data = {'y': flat_list, 'x': flat_list_x}
        df_plot = pd.DataFrame(data)

        sns.set_theme(style='white')

        if label==0:
            get_plot = sns.lineplot(x = 'x', y='y', data=df_plot, ax= axs1,\
                            color='#6495ED'
                          )
            axs1.set(ylim=(-5,5))
            axs1.set_title("Cluster "+str(label))
            axs1.set_ylabel('')
            axs1.set_xlabel('')            
            fig1.supylabel('Value of the measurement (scaled)')
            fig1.supxlabel('Age at measurement (weeks)')
            fig1.tight_layout()


        if label==1:
            get_plot = sns.lineplot(x = 'x', y='y', data=df_plot, ax= axs2,\
                            color='#DA70D6'
                           )
            axs2.set(ylim=(-5,5))

            axs2.set_title("Cluster "+str(label))
            axs2.set_ylabel('')
            axs2.set_xlabel('')

            fig2.supylabel('Value of the measurement (scaled)')
            fig2.supxlabel('Age at measurement (weeks)')
            fig2.tight_layout()


        if label==2:
            get_plot = sns.lineplot(x = 'x', y='y', data=df_plot, ax= axs3,\
                            color="#20B2AA"
                           )
            axs3.set(ylim=(-5,5))

            axs3.set_title("Cluster "+str(label))
            axs3.set_ylabel('')
            axs3.set_xlabel('')

            fig3.supylabel('Value of the measurement (scaled)')
            fig3.supxlabel('Age at measurement (weeks)')
            fig3.tight_layout()

        if label==3:
            get_plot = sns.lineplot(x = 'x', y='y', data=df_plot, ax= axs4,\
                            color="#FF8C00"
                           )
            axs4.set(ylim=(-5,5))

            axs4.set_title("Cluster "+str(label))
            axs4.set_ylabel('')
            axs4.set_xlabel('')
            fig4.supylabel('Value of the measurement (scaled)')
            fig4.supxlabel('Age at measurement (weeks)')
            fig4.tight_layout()

        if label==4:
            get_plot = sns.lineplot(x = 'x', y='y', data=df_plot, ax= axs5,\
                            color="red"
                           )
            axs5.set(ylim=(-5,5))

            axs5.set_title("Cluster "+str(label))
            axs5.set_ylabel('')
            axs5.set_xlabel('')
            fig5.supylabel('Value of the measurement (scaled)')
            fig5.supxlabel('Age at measurement (weeks)')
            fig5.tight_layout()


fig1.savefig("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Figures/measures_in_cluster0_lineplot_shome.pdf")
fig2.savefig("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Figures/measures_in_cluster1_lineplot_shome.pdf")
fig3.savefig("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Figures/measures_in_cluster2_lineplot_shome.pdf")
fig4.savefig("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Figures/measures_in_cluster3_lineplot_shome.pdf")
fig5.savefig("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Figures/measures_in_cluster4_lineplot_shome.pdf")

fig1.savefig("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Figures/measures_in_cluster0_lineplot_shome.png", dpi=300)
fig2.savefig("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Figures/measures_in_cluster1_lineplot_shome.png", dpi=300)
fig3.savefig("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Figures/measures_in_cluster2_lineplot_shome.png", dpi=300)
fig4.savefig("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Figures/measures_in_cluster3_lineplot_shome.png", dpi=300)
fig5.savefig("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Figures/measures_in_cluster4_lineplot_shome.png", dpi=300)


# In[7]:


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





# In[ ]:


fig.savefig("/Users/sshome/Desktop/Bioinformatics_projects/Neelu_project/Neonatal_measurements_final_share/scripts/Neonatal_measurements_final/Figures/WBC_color_coded_basedOn_clusters.pdf")


# In[12]:


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
    print(nameofSeries)



fig, axs = plt.subplots(2,3,layout='constrained',figsize=(30,15))
# palette = itertools.cycle(sns.color_palette('Set3'))
# palette = itertools.cycle(sns.color_palette('Set2'))

axs = axs.ravel()

# For each label there is,
# plots every series with that label

for label,color in zip(set(labels),colors):
    cluster = []
    var_names = []
    vars_of_int = ['Monocytes [#/volume] in Blood by Manual count',\
                   'Monocytes [#/volume] in Body fluid by Manual count',\
                    'Monocytes/100 leukocytes in Blood',\
                    'Monocytes/100 leukocytes in Body fluid',\
                   'Monocytes/100 leukocytes in Cerebral spinal fluid'\
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





# In[10]:


labels


# In[ ]:




