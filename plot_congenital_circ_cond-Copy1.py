#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reset', '')
#from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import regex as re


# In[2]:


pip install bigquery


# In[4]:


# baby_measures = pd.read_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/PheWAS_babiesOver15_withCBC/analysis/disease_new_multivariate_icd9includ/measurements_with_cong_cond_label.csv")
baby_cong_labels = pd.read_csv("/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/congenital_circ_labels.csv")
baby_measures = pd.read_csv("/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv")
baby_cong_labels = baby_cong_labels.groupby(['person_id']).agg('max').reset_index()
cases_id = baby_cong_labels[baby_cong_labels['label']==1].person_id.unique()
baby_measures['label'] = baby_measures['person_id'].isin(cases_id).astype(int)


# In[6]:


import pandas as pd

# Load data
baby_cong_labels = pd.read_csv("/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/congenital_circ_labels.csv")
baby_measures = pd.read_csv("/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv")

# Ensure label is numeric
baby_cong_labels['label'] = pd.to_numeric(baby_cong_labels['label'], errors='coerce')

# Group and get max label for each person_id
baby_cong_labels = baby_cong_labels.groupby('person_id', as_index=False)['label'].max()

# Get list of cases with label == 1
cases_id = baby_cong_labels.loc[baby_cong_labels['label'] == 1, 'person_id'].unique()

# Add label to baby_measures
baby_measures['label'] = baby_measures['person_id'].isin(cases_id).astype(int)


# In[7]:


meas_of_int = ['Neutrophils/100 leukocytes in Blood', 'Monocytes/100 leukocytes in Blood'\
              ,"Lymphocytes/100 leukocytes in Blood", "Basophils/100 leukocytes in Blood", \
               "Eosinophils/100 leukocytes in Blood"]
baby_measures_int = baby_measures[['person_id', meas_of_int[0], meas_of_int[1],meas_of_int[2],meas_of_int[3],\
                                   meas_of_int[4],'age_at_meas','label']]
df_quant_neut = baby_measures_int[['person_id','age_at_meas', meas_of_int[0],'label']]
df_quant_mono = baby_measures_int[['person_id', 'age_at_meas', meas_of_int[1],'label']]
df_quant_lymph = baby_measures_int[['person_id', 'age_at_meas', meas_of_int[2],'label']]
df_quant_baso = baby_measures_int[['person_id', 'age_at_meas', meas_of_int[3],'label']]
df_quant_eosi = baby_measures_int[['person_id', 'age_at_meas', meas_of_int[4],'label']]


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import regex as re
##--- plotting all together
fig7, axs = plt.subplots(5,1,figsize=(20,30), sharex=True)
axs = axs.ravel()
sns.set_theme(style='white')
medianprops = dict(color="black",linewidth=1.5)
my_colors = ['cornflowerblue',"salmon"]
sns.set_palette( my_colors )



l2 = sns.lineplot(x=df_quant_neut['age_at_meas'], y=df_quant_neut['Neutrophils/100 leukocytes in Blood'], hue \
                      =df_quant_neut['label'], ax=axs[0])
l4 = sns.lineplot(x=df_quant_baso['age_at_meas'], y=df_quant_baso["Basophils/100 leukocytes in Blood"], hue \
                      =df_quant_baso['label'], ax=axs[1])
l6 = sns.lineplot(x=df_quant_lymph['age_at_meas'], y=df_quant_lymph["Lymphocytes/100 leukocytes in Blood"], hue \
                      =df_quant_lymph['label'], ax=axs[2])
l8 = sns.lineplot(x=df_quant_eosi['age_at_meas'], y=df_quant_eosi["Eosinophils/100 leukocytes in Blood"], hue \
                      =df_quant_eosi['label'], ax=axs[3])
l10 = sns.lineplot(x=df_quant_mono['age_at_meas'], y=df_quant_mono["Monocytes/100 leukocytes in Blood"], hue \
                      =df_quant_mono['label'], ax=axs[4])

legend = axs[0].legend(fontsize = 20)
legend.texts[0].set_text("")
axs[0].get_legend().get_texts()[0].set_text('No Cong. Anom. of Circ. Sys.')
axs[0].get_legend().get_texts()[1].set_text('Cong. Anom. of Circ. Sys.')
axs[1].get_legend().remove()
axs[2].get_legend().remove()
axs[3].get_legend().remove()
axs[4].get_legend().remove()

axs[0].set_ylabel("Neutrophils",fontsize=40)
axs[1].set_ylabel("Basophils",fontsize=40)
axs[2].set_ylabel("Lymphocytes",fontsize=40)
axs[3].set_ylabel("Eosinophils",fontsize=40)
axs[4].set_ylabel("Monocytes",fontsize=40)


axs[0].set_xlabel("")
axs[1].set_xlabel("")
axs[2].set_xlabel("")
axs[3].set_xlabel("")
axs[4].set_xlabel("Age at measurement (weeks)")

axs[0].xaxis.set_tick_params(labelbottom=True)
axs[1].xaxis.set_tick_params(labelbottom=True)
axs[2].xaxis.set_tick_params(labelbottom=True)
axs[3].xaxis.set_tick_params(labelbottom=True)
axs[4].xaxis.set_tick_params(labelbottom=True)




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





fig7.tight_layout()


fig7.savefig("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Figures/allWBC_trend_congenital_heart.png", dpi=300)
fig7.savefig("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Figures/allWBC_trend_congenital_heart.pdf")


# In[25]:


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

# === Load data ===
baby_cong_labels = pd.read_csv("/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/congenital_circ_labels.csv")
baby_measures = pd.read_csv("/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv")

# === Prepare data ===
baby_cong_labels['label'] = pd.to_numeric(baby_cong_labels['label'], errors='coerce')
baby_cong_labels = baby_cong_labels.groupby('person_id', as_index=False)['label'].max()

cases_id = baby_cong_labels.loc[baby_cong_labels['label'] == 1, 'person_id'].unique()
baby_measures['label'] = baby_measures['person_id'].isin(cases_id).astype(int)

meas_of_int = [
      'Neutrophils/100 leukocytes in Blood',
    'Basophils/100 leukocytes in Blood',
    'Lymphocytes/100 leukocytes in Blood',
    'Eosinophils/100 leukocytes in Blood',
    'Monocytes/100 leukocytes in Blood',  
]

# Keep only relevant columns
baby_measures_int = baby_measures[['person_id', 'age_at_meas'] + meas_of_int + ['label']]

# === Define time bins (e.g., 0-4, 4-8, 8-12, 12-16 weeks) ===
bins = [0, 4, 8, 12, 16, 20, 26]
labels = ['0-4w', '4-8w', '8-12w', '12-16w', '16-20w', '20-26w']
baby_measures_int['age_bin'] = pd.cut(baby_measures_int['age_at_meas'], bins=bins, labels=labels, right=False)

# Drop rows without bin assignment
baby_measures_int = baby_measures_int.dropna(subset=['age_bin'])

# === Compute means and run t-tests ===
results = []
for meas in meas_of_int:
    for age_group in labels:
        subset = baby_measures_int[baby_measures_int['age_bin'] == age_group]
        group0 = subset.loc[subset['label'] == 0, meas].dropna()
        group1 = subset.loc[subset['label'] == 1, meas].dropna()

        if len(group0) > 1 and len(group1) > 1:
            t_stat, p_val = ttest_ind(group0, group1, equal_var=False, nan_policy='omit')
        else:
            t_stat, p_val = np.nan, np.nan

        results.append({
            'Marker': meas,
            'Age_Group': age_group,
            'Mean_No_Condition': group0.mean() if len(group0) > 0 else np.nan,
            'Mean_Condition': group1.mean() if len(group1) > 0 else np.nan,
            'n_No_Condition': len(group0),
            'n_Condition': len(group1),
            't_stat': t_stat,
            'p_val': p_val
        })

results_df = pd.DataFrame(results)

# === Sort and display ===
results_df = results_df.sort_values(by=['Marker', 'Age_Group'])
pd.set_option('display.float_format', lambda x: '%.3f' % x)

print("\n=== T-Test Results by Marker and Age Group ===\n")
print(results_df)

# === Visualization (optional, updated with bins) ===
sns.set_theme(style='white')
my_colors = ['cornflowerblue', "salmon"]
sns.set_palette(my_colors)

fig, axs = plt.subplots(len(meas_of_int), 1, figsize=(20, 30), sharex=True)
axs = axs.ravel()

for i, meas in enumerate(meas_of_int):
    sns.lineplot(x='age_at_meas', y=meas, hue='label', data=baby_measures_int, ax=axs[i])
    axs[i].set_ylabel(meas.split('/')[0], fontsize=30)
    axs[i].tick_params(axis='both', labelsize=16)

axs[-1].set_xlabel("Age at measurement (weeks)", fontsize=30)
fig.tight_layout()
plt.show()


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Assume results_df is already loaded with columns: Marker, Age_Group, p_val

# === Custom sorting orders ===
# Age group sorting
def extract_start_week(age_group):
    return int(age_group.split('-')[0].replace('w', ''))

results_df['Age_Sort'] = results_df['Age_Group'].apply(extract_start_week)
results_df = results_df.sort_values(by='Age_Sort')

# Marker sorting based on provided order from image
custom_marker_order = [
    'Neutrophils/100 leukocytes in Blood',
    'Basophils/100 leukocytes in Blood',
    'Lymphocytes/100 leukocytes in Blood',
    'Eosinophils/100 leukocytes in Blood',
    'Monocytes/100 leukocytes in Blood'
]

# Pivot table
heatmap_data = results_df.pivot(index='Marker', columns='Age_Group', values='p_val')

# Sort columns by age
sorted_age_groups = sorted(results_df['Age_Group'].unique(), key=lambda x: int(x.split('-')[0].replace('w', '')))
heatmap_data = heatmap_data[sorted_age_groups]

# Reorder rows by custom marker order
heatmap_data = heatmap_data.reindex(custom_marker_order)

# Convert p-values to -log10 scale
heatmap_data_log = -np.log10(heatmap_data)

# Annotate significance
annot = heatmap_data.applymap(lambda x: "*" if (pd.notnull(x) and x < 0.05) else "")

# === Create custom blue-gray-pink colormap ===
colors = ["#2166AC", "#D9D9D9", "#B2182B"]
custom_cmap = LinearSegmentedColormap.from_list("blue_gray_pink", colors, N=256)

# === Plot heatmap ===
plt.figure(figsize=(14, 8))
sns.set_theme(style="white")

ax = sns.heatmap(
    heatmap_data_log,
    cmap=custom_cmap,
    center=0,
    annot=annot,
    fmt="",
    linewidths=0.6,
    linecolor="white",
    cbar_kws={"label": "-log10(p-value)"},
    vmin=0,
    vmax=np.nanmax(heatmap_data_log)
)

# Title and axis labels
ax.set_title("Significance Heatmap (-log10 p-values)", fontsize=20, weight='bold', pad=20)
ax.set_xlabel("Age Group (Weeks)", fontsize=16, weight='bold', labelpad=10)
ax.set_ylabel("Marker", fontsize=16, weight='bold', labelpad=10)

# Customize ticks
ax.tick_params(axis='x', labelrotation=45, labelsize=14)
ax.tick_params(axis='y', labelsize=14)

# Bold tick labels
for tick in ax.get_xticklabels():
    tick.set_weight('bold')
for tick in ax.get_yticklabels():
    tick.set_weight('bold')

# Frame enhancement
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(1)

plt.tight_layout()

# Save high-resolution image
plt.savefig("significance_heatmap_blue_gray_pink_sorted_by_marker_and_age.png", dpi=300, bbox_inches='tight')
plt.show()


# In[27]:


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.stats.multitest import multipletests

# === Load data ===
baby_cong_labels = pd.read_csv("/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/congenital_circ_labels.csv")
baby_measures = pd.read_csv("/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv")

# === Prepare data ===
baby_cong_labels['label'] = pd.to_numeric(baby_cong_labels['label'], errors='coerce')
baby_cong_labels = baby_cong_labels.groupby('person_id', as_index=False)['label'].max()

cases_id = baby_cong_labels.loc[baby_cong_labels['label'] == 1, 'person_id'].unique()
baby_measures['label'] = baby_measures['person_id'].isin(cases_id).astype(int)

# WBC markers of interest
meas_of_int = [
    'Neutrophils/100 leukocytes in Blood',
    'Basophils/100 leukocytes in Blood',
    'Lymphocytes/100 leukocytes in Blood',
    'Eosinophils/100 leukocytes in Blood',
    'Monocytes/100 leukocytes in Blood'
]

# Keep only relevant columns
baby_measures_int = baby_measures[['person_id', 'age_at_meas'] + meas_of_int + ['label']]

# === Define time bins ===
bins = [0, 4, 8, 12, 16, 20, 26]
labels = ['0-4w', '4-8w', '8-12w', '12-16w', '16-20w', '20-26w']
baby_measures_int['age_bin'] = pd.cut(baby_measures_int['age_at_meas'], bins=bins, labels=labels, right=False)

# Drop rows without bin assignment
baby_measures_int = baby_measures_int.dropna(subset=['age_bin'])

# === Compute Welch's t-tests ===
results = []
for meas in meas_of_int:
    for age_group in labels:
        subset = baby_measures_int[baby_measures_int['age_bin'] == age_group]
        group0 = subset.loc[subset['label'] == 0, meas].dropna()
        group1 = subset.loc[subset['label'] == 1, meas].dropna()

        if len(group0) > 1 and len(group1) > 1:
            t_stat, p_val = ttest_ind(group0, group1, equal_var=False, nan_policy='omit')
        else:
            t_stat, p_val = np.nan, np.nan

        results.append({
            'Marker': meas,
            'Age_Group': age_group,
            'Mean_No_Condition': group0.mean() if len(group0) > 0 else np.nan,
            'Mean_Condition': group1.mean() if len(group1) > 0 else np.nan,
            'n_No_Condition': len(group0),
            'n_Condition': len(group1),
            't_stat': t_stat,
            'p_val': p_val
        })

results_df = pd.DataFrame(results)

# === Apply FDR correction (Benjamini-Hochberg) ===
mask_valid = results_df['p_val'].notna()
pvals = results_df.loc[mask_valid, 'p_val']
rejected, p_adj, _, _ = multipletests(pvals, method='fdr_bh')
results_df.loc[mask_valid, 'p_adj'] = p_adj
results_df.loc[mask_valid, 'significant'] = rejected

# === Prepare data for heatmap ===
# Sort by Age and Marker
def extract_start_week(age_group):
    return int(age_group.split('-')[0].replace('w', ''))

results_df['Age_Sort'] = results_df['Age_Group'].apply(extract_start_week)
results_df = results_df.sort_values(by=['Marker', 'Age_Sort'])

# Custom marker order
custom_marker_order = [
    'Neutrophils/100 leukocytes in Blood',
    'Basophils/100 leukocytes in Blood',
    'Lymphocytes/100 leukocytes in Blood',
    'Eosinophils/100 leukocytes in Blood',
    'Monocytes/100 leukocytes in Blood'
]

# Pivot for heatmap
heatmap_data = results_df.pivot(index='Marker', columns='Age_Group', values='p_adj')  # use adjusted p-values
sorted_age_groups = sorted(labels, key=lambda x: int(x.split('-')[0].replace('w', '')))
heatmap_data = heatmap_data[sorted_age_groups]
heatmap_data = heatmap_data.reindex(custom_marker_order)

# Convert to -log10 scale
heatmap_data_log = -np.log10(heatmap_data)

# Annotate: show * if FDR-adjusted p < 0.05
annot = heatmap_data.applymap(lambda x: "*" if (pd.notnull(x) and x < 0.05) else "")

# === Plot heatmap ===
colors = ["#2166AC", "#D9D9D9", "#B2182B"]  # blue-gray-pink
custom_cmap = LinearSegmentedColormap.from_list("blue_gray_pink", colors, N=256)

plt.figure(figsize=(14, 8))
sns.set_theme(style="white")

ax = sns.heatmap(
    heatmap_data_log,
    cmap=custom_cmap,
    annot=annot,
    fmt="",
    linewidths=0.6,
    linecolor="white",
    cbar_kws={"label": "-log10(FDR-adjusted p-value)"},
    vmin=0,
    vmax=np.nanmax(heatmap_data_log)
)

ax.set_title("Significance Heatmap (-log10 FDR-adjusted p-values): CHD vs Non-CHD", fontsize=20, weight='bold', pad=20)
ax.set_xlabel("Age Group (Weeks)", fontsize=16, weight='bold', labelpad=10)
ax.set_ylabel("Marker", fontsize=16, weight='bold', labelpad=10)

ax.tick_params(axis='x', labelrotation=45, labelsize=14)
ax.tick_params(axis='y', labelsize=14)

for tick in ax.get_xticklabels():
    tick.set_weight('bold')
for tick in ax.get_yticklabels():
    tick.set_weight('bold')

for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(1)

plt.tight_layout()
plt.savefig("chd_vs_nonchd_significance_heatmap_fdr_corrected.png", dpi=300, bbox_inches='tight')
plt.show()


# In[28]:


# === Plot heatmap with bigger asterisks ===
colors = ["#2166AC", "#D9D9D9", "#B2182B"]  # blue-gray-pink
custom_cmap = LinearSegmentedColormap.from_list("blue_gray_pink", colors, N=256)

plt.figure(figsize=(14, 8))
sns.set_theme(style="white")

ax = sns.heatmap(
    heatmap_data_log,
    cmap=custom_cmap,
    annot=annot,
    fmt="",
    annot_kws={"size": 18, "weight": "bold"},  # <-- Make asterisks bigger & bold
    linewidths=0.6,
    linecolor="white",
    cbar_kws={"label": "-log10(FDR-adjusted p-value)"},
    vmin=0,
    vmax=np.nanmax(heatmap_data_log)
)

# Title and labels
ax.set_title("Significance Heatmap (-log10 FDR-adjusted p-values): CHD vs Non-CHD", fontsize=20, weight='bold', pad=20)
ax.set_xlabel("Age Group (Weeks)", fontsize=16, weight='bold', labelpad=10)
ax.set_ylabel("Marker", fontsize=16, weight='bold', labelpad=10)

# Customize ticks
ax.tick_params(axis='x', labelrotation=45, labelsize=14)
ax.tick_params(axis='y', labelsize=14)

for tick in ax.get_xticklabels():
    tick.set_weight('bold')
for tick in ax.get_yticklabels():
    tick.set_weight('bold')

# Frame enhancement
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(1)

plt.tight_layout()
plt.savefig("chd_vs_nonchd_significance_heatmap_fdr_corrected_big_asterisks.png", dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:




