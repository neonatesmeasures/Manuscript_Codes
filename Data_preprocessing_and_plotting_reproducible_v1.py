#!/usr/bin/env python
# coding: utf-8

# In[1]:


#loading required packages
get_ipython().run_line_magic('reset', '')
#from google.cloud import bigquery
import os
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import nanmean


# Import actual data from the OMOP as "df_measures_outlier_rmv" dataframe 
The next code will be helpful to format the values as follows :

If there are multiple measurement_concept_ids for each measurement name, they are included under one variable. For example, if time1 from automated count has a recording, then time2 from manual count has a recording and they are sorted (based on time) and the two readings are treated as if they were one variable. In case there are more values recorded for each time-point, the values are averaged to plot something similar to a time series.
# In[2]:


df_measures_units = pd.read_csv("/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv")


# In[3]:


#plotting the figures with each lab measurement

test = df_measures_units[['Neutrophils/100 leukocytes in Blood', 'age_at_meas']]
sns.lineplot(data= test, x = 'age_at_meas', y = 'Neutrophils/100 leukocytes in Blood' )


# In[4]:


test = df_measures_units[['Basophils/100 leukocytes in Blood', 'age_at_meas']]
sns.lineplot(data= test, x = 'age_at_meas', y = 'Basophils/100 leukocytes in Blood' )


# In[4]:


test = df_measures_units[['Eosinophils/100 leukocytes in Blood', 'age_at_meas']]
sns.lineplot(data= test, x = 'age_at_meas', y = 'Eosinophils/100 leukocytes in Blood' )


# In[15]:


import numpy as np, scipy
print("NumPy:", np.__version__, "| SciPy:", scipy.__version__)
from scipy.stats import linregress
print("linregress OK")


# In[9]:


# --- SciPy import diagnostics (run in Jupyter) ---
import sys, os, pathlib, importlib, pkgutil

def scan_for_shadowing(root, names=("scipy.py","scipy")):
    root = pathlib.Path(root)
    hits = []
    for name in names:
        p = root / name
        if p.exists():
            hits.append(str(p.resolve()))
    return hits

print("PYTHON EXECUTABLE:", sys.executable)
print("WORKING DIR:", os.getcwd())
print("sys.path[0] (where shadowing often happens):", sys.path[0])

# Try to import whatever "scipy" currently resolves to
import scipy
print("Imported 'scipy' object type:", type(scipy))
print("scipy module attributes contain '__version__'? ", hasattr(scipy, "__version__"))

scipy_file = getattr(scipy, "__file__", None)
print("scipy.__file__ ->", scipy_file)

# Look for local shadowing next to your notebook
hits = scan_for_shadowing(os.getcwd())
if hits:
    print("\nPOSSIBLE SHADOWING IN CURRENT DIR:")
    for h in hits:
        print("  -", h)

# Also check immediate parent of the notebook (common when using src/ or project/ layouts)
parent_hits = scan_for_shadowing(pathlib.Path(os.getcwd()).parent)
if parent_hits:
    print("\nPOSSIBLE SHADOWING IN PARENT DIR:")
    for h in parent_hits:
        print("  -", h)

# Show a few first sys.path entries to see precedence
print("\nTop of sys.path (import precedence):")
for p in sys.path[:5]:
    print("  -", p)


# In[11]:


get_ipython().system('pip install --no-cache-dir --upgrade scipy==1.13.1')


# In[16]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

# -----------------------------
# File paths (edit if needed)
# -----------------------------
file_path1 = "/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv"
file_path2 = "/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/neelu_validation_stuffs/day_summary_for_stanford.csv"

# -----------------------------
# Load data with basic validation
# -----------------------------
if not os.path.exists(file_path1):
    raise FileNotFoundError(f"Missing file_path1: {file_path1}")
if not os.path.exists(file_path2):
    raise FileNotFoundError(f"Missing file_path2: {file_path2}")

df_measures_units = pd.read_csv(file_path1)
df = pd.read_csv(file_path2)

# If the second CSV has 5 columns but different names, standardize them
if df.shape[1] >= 5:
    df = df.iloc[:, :5]  # keep first 5 cols if extras present
    df.columns = ['Unnamed', 'age_at_meas', 'measurement', 'mean_val', 'stdev']
else:
    expected = ['Unnamed', 'age_at_meas', 'measurement', 'mean_val', 'stdev']
    raise ValueError(f"Expected at least 5 columns in file_path2. Saw {df.columns.tolist()}")

# Convert age in days -> weeks for dataset 2 (UCSF-style table)
df['age_at_meas'] = df['age_at_meas'] / 7.0
# normalize measurement names
df['measurement'] = df['measurement'].astype(str).str.strip()

# -----------------------------
# Config
# -----------------------------
cell_types = ['Neutrophils', 'Eosinophils', 'Lymphocytes', 'Monocytes', 'Basophils']

# Use half-open intervals [start, end) to avoid double-counting edges
intervals = [(1, 7), (7, 13), (13, 19), (19, 25)]

# -----------------------------
# Helpers
# -----------------------------
def calculate_interval_means(df_in, xcol, ycol, intervals_):
    """Return means of ycol within [start, end) on xcol for each interval."""
    means = []
    for start, end in intervals_:
        m = df_in[(df_in[xcol] >= start) & (df_in[xcol] < end)]
        means.append(m[ycol].mean() if not m.empty else np.nan)
    return means

def annotate_slope(ax, xvals, yvals, color, start, end):
    """Fit slope in interval and annotate at the midpoint."""
    # ensure we have numeric arrays without NaNs
    xy = pd.DataFrame({'x': pd.to_numeric(xvals, errors='coerce'),
                       'y': pd.to_numeric(yvals, errors='coerce')}).dropna()
    if len(xy) > 1:
        slope, intercept, r, p, se = linregress(xy['x'], xy['y'])
        midpoint = (start + end) / 2
        # Compute y position 10% above axis bottom
        ymin, ymax = ax.get_ylim()
        y_pos = ymin + 0.1 * (ymax - ymin)
        ax.annotate(f'Slope: {slope:.2f}', xy=(midpoint, y_pos),
                    color=color, ha='center', va='bottom')
        return slope
    return None

# -----------------------------
# Plot
# -----------------------------
n_rows = len(cell_types)
fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4.5 * n_rows), squeeze=False)

trend_line_color = 'gray'
line_plot_color = '#2ca02c'  # green

for i, cell_type in enumerate(cell_types):
    # Stanford-style wide table expects columns like "Neutrophils/100 leukocytes in Blood"
    col_name = f'{cell_type}/100 leukocytes in Blood'
    ax_left = axes[i, 0]
    ax_right = axes[i, 1]

    # ---------- Left plot: Stanford (df_measures_units) ----------
    if 'age_at_meas' in df_measures_units.columns and col_name in df_measures_units.columns:
        st = df_measures_units[['age_at_meas', col_name]].dropna()
        # interval means
        interval_means = calculate_interval_means(st, 'age_at_meas', col_name, intervals)
        midpoints = [(a + b) / 2 for a, b in intervals]

        # bars + line on mean-of-intervals
        ax_left.bar(midpoints, interval_means, width=[(b - a) * 0.8 for a, b in intervals],
                    color='lightblue', alpha=0.5)
        ax_left.plot(midpoints, interval_means, color=trend_line_color, marker='o',
                     linestyle='-', linewidth=2, markersize=6)

        # overlay raw line plot
        sns.lineplot(data=st, x='age_at_meas', y=col_name, ax=ax_left,
                     errorbar=None, color=line_plot_color)

        ax_left.set_title(f'{cell_type}/100 leukocytes in Blood - Stanford')
        ax_left.set_xlabel('Age at Measurement (weeks)')
        ax_left.set_ylabel(f'{cell_type}/100 leukocytes in Blood')

        # vertical guides and slope annotations
        for (start, end) in intervals:
            segment = st[(st['age_at_meas'] >= start) & (st['age_at_meas'] < end)]
            if not segment.empty:
                slope = annotate_slope(ax_left, segment['age_at_meas'], segment[col_name],
                                       trend_line_color, start, end)
                ax_left.axvline(x=start, linestyle='--', color='gray', alpha=0.4)
                ax_left.axvline(x=end, linestyle='--', color='gray', alpha=0.4)
    else:
        ax_left.text(0.5, 0.5,
                     f"Missing columns for Stanford plot:\n'required: age_at_meas and {col_name}'",
                     ha='center', va='center', transform=ax_left.transAxes)
        ax_left.set_title(f'{cell_type} - Stanford (missing data)')
        ax_left.set_xticks([]); ax_left.set_yticks([])

    # ---------- Right plot: UCSF (df) ----------
    if all(c in df.columns for c in ['age_at_meas', 'measurement', 'mean_val']):
        dff = df[df['measurement'].str.lower() == cell_type.lower()].copy()
        if not dff.empty:
            # interval means
            interval_means_u = calculate_interval_means(dff, 'age_at_meas', 'mean_val', intervals)
            midpoints = [(a + b) / 2 for a, b in intervals]

            ax_right.bar(midpoints, interval_means_u, width=[(b - a) * 0.8 for a, b in intervals],
                         color='lightblue', alpha=0.5)
            ax_right.plot(midpoints, interval_means_u, color=trend_line_color, marker='o',
                          linestyle='-', linewidth=2, markersize=6)

            sns.lineplot(data=dff, x='age_at_meas', y='mean_val', ax=ax_right,
                         errorbar=None, color=line_plot_color)

            ax_right.set_title(f'{cell_type}/100 leukocytes in Blood - UCSF')
            ax_right.set_xlabel('Age at Measurement (weeks)')
            ax_right.set_ylabel(f'{cell_type}/100 leukocytes in Blood')

            for (start, end) in intervals:
                seg = dff[(dff['age_at_meas'] >= start) & (dff['age_at_meas'] < end)]
                if not seg.empty:
                    slope = annotate_slope(ax_right, seg['age_at_meas'], seg['mean_val'],
                                           trend_line_color, start, end)
                    ax_right.axvline(x=start, linestyle='--', color='gray', alpha=0.4)
                    ax_right.axvline(x=end, linestyle='--', color='gray', alpha=0.4)
        else:
            ax_right.text(0.5, 0.5, f"No UCSF rows for '{cell_type}'",
                          ha='center', va='center', transform=ax_right.transAxes)
            ax_right.set_title(f'{cell_type} - UCSF (no rows)')
            ax_right.set_xticks([]); ax_right.set_yticks([])
    else:
        ax_right.text(0.5, 0.5,
                      "Missing columns for UCSF plot:\nrequired: age_at_meas, measurement, mean_val",
                      ha='center', va='center', transform=ax_right.transAxes)
        ax_right.set_title(f'{cell_type} - UCSF (missing data)')
        ax_right.set_xticks([]); ax_right.set_yticks([])

plt.tight_layout()
plt.savefig('cohort_cbc_comparison_version2.png', dpi=300)
plt.show()
print("Saved figure -> cohort_cbc_comparison_version2.png")


# In[17]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load the data
file_path1 = "/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv"
df_measures_units = pd.read_csv(file_path1)

file_path2 = '/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/neelu_validation_stuffs/day_summary_for_stanford.csv'
df = pd.read_csv(file_path2)
df.columns = ['Unnamed', 'age_at_meas', 'measurement', 'mean_val', 'stdev']

# Convert age_at_meas in dataset 2 from days to weeks
df['age_at_meas'] = df['age_at_meas'] / 7

# Cell types to compare
cell_types = ['Neutrophils', 'Eosinophils', 'Lymphocytes', 'Monocytes','Basophils']

# Define intervals
intervals = [(1, 7), (7, 13), (13, 19), (19, 25)]

# Function to calculate mean values for intervals
def calculate_interval_means(df, col_name, intervals):
    means = []
    for start, end in intervals:
        interval = df[(df['age_at_meas'] >= start) & (df['age_at_meas'] <= end)]
        if not interval.empty:
            means.append(interval[col_name].mean())
        else:
            means.append(np.nan)
    return means

# Function to plot regression lines and annotate slope
def annotate_slope(ax, x, y, color, start, end):
    if len(x) > 1 and len(y) > 1:
        slope, intercept, _, _, _ = linregress(x, y)
        midpoint = (start + end) / 2
        y_pos = ax.get_ylim()[0] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.annotate(f'Slope: {slope:.2f}', xy=(midpoint, y_pos), color=color, ha='center')
        return slope
    else:
        print(f"Insufficient data points to calculate slope: {len(x)} points")
        return None

# Create subplots
fig, axes = plt.subplots(len(cell_types), 2, figsize=(15, 25))

trend_line_color = 'gray'
line_plot_color = '#2ca02c'  # Green color

for i, cell_type in enumerate(cell_types):
    col_name = f'{cell_type}/100 leukocytes in Blood'

    # Plot Stanford data
    if col_name in df_measures_units.columns:
        test = df_measures_units[['age_at_meas', col_name]].dropna()

        # Calculate mean values for intervals
        interval_means = calculate_interval_means(test, col_name, intervals)
        interval_midpoints = [(start + end) / 2 for start, end in intervals]

        # Create bar plot for interval means
        axes[i, 0].bar(interval_midpoints, interval_means, width=4, color='lightblue', alpha=0.5)

        # Add trend line connecting the midpoints
        axes[i, 0].plot(interval_midpoints, interval_means, color=trend_line_color, marker='o', linestyle='-', linewidth=2, markersize=6)

        # Overlay line plot
        sns.lineplot(data=test, x='age_at_meas', y=col_name, ax=axes[i, 0], ci=None, color=line_plot_color)

        axes[i, 0].set_title(f'{cell_type}/100 leukocytes in Blood - Stanford')
        axes[i, 0].set_xlabel('Age at Measurement (weeks)')
        axes[i, 0].set_ylabel(f'{cell_type}/100 leukocytes in Blood')
        axes[i, 0].tick_params(axis='x')

        # Annotate slopes for Stanford data
        for j, (start, end) in enumerate(intervals):
            interval = test[(test['age_at_meas'] >= start) & (test['age_at_meas'] <= end)]
            if not interval.empty:
                print(f"Processing {cell_type} for Stanford: Interval {start}-{end} weeks, {len(interval)} points")
                print(interval[['age_at_meas', col_name]])  # Debugging: Check interval data
                slope = annotate_slope(axes[i, 0], interval['age_at_meas'], interval[col_name], trend_line_color, start, end)
                if slope is not None:
                    axes[i, 0].axvline(x=start, linestyle='--', color='gray', alpha=0.5)
                    axes[i, 0].axvline(x=end, linestyle='--', color='gray', alpha=0.5)
                    print(f'{cell_type} - Stanford Slope ({start}-{end} weeks): {slope}')
            else:
                print(f"No data for {cell_type} in Stanford: Interval {start}-{end} weeks")

    # Plot UCSF data
    df_filtered = df[df['measurement'].str.lower() == cell_type.lower()]
    if not df_filtered.empty:
        # Calculate mean values for intervals
        interval_means = calculate_interval_means(df_filtered, 'mean_val', intervals)
        interval_midpoints = [(start + end) / 2 for start, end in intervals]

        # Create bar plot for interval means
        axes[i, 1].bar(interval_midpoints, interval_means, width=4, color='lightblue', alpha=0.5)

        # Add trend line connecting the midpoints
        axes[i, 1].plot(interval_midpoints, interval_means, color=trend_line_color, marker='o', linestyle='-', linewidth=2, markersize=6)

        # Overlay line plot
        sns.lineplot(data=df_filtered, x='age_at_meas', y='mean_val', ax=axes[i, 1], ci=None, color=line_plot_color)

        axes[i, 1].set_title(f'{cell_type}/100 leukocytes in Blood - UCSF')
        axes[i, 1].set_xlabel('Age at Measurement (weeks)')
        axes[i, 1].set_ylabel(f'{cell_type}/100 leukocytes in Blood')
        axes[i, 1].tick_params(axis='x')

        # Annotate slopes for UCSF data
        for j, (start, end) in enumerate(intervals):
            interval = df_filtered[(df_filtered['age_at_meas'] >= start) & (df_filtered['age_at_meas'] <= end)]
            if not interval.empty:
                print(f"Processing {cell_type} for UCSF: Interval {start}-{end} weeks, {len(interval)} points")
                print(interval[['age_at_meas', 'mean_val']])  # Debugging: Check interval data
                slope = annotate_slope(axes[i, 1], interval['age_at_meas'], interval['mean_val'], trend_line_color, start, end)
                if slope is not None:
                    axes[i, 1].axvline(x=start, linestyle='--', color='gray', alpha=0.5)
                    axes[i, 1].axvline(x=end, linestyle='--', color='gray', alpha=0.5)
                    print(f'{cell_type} - UCSF Slope ({start}-{end} weeks): {slope}')
            else:
                print(f"No data for {cell_type} in UCSF: Interval {start}-{end} weeks")

plt.tight_layout()
plt.savefig('cohort_cbc_comparison_version2.png', dpi=300)
plt.show()


# In[18]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, wilcoxon
import warnings

warnings.filterwarnings("ignore")

# Load data
file_path1 = "/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv"
df_stanford = pd.read_csv(file_path1)

file_path2 = '/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/neelu_validation_stuffs/day_summary_for_stanford.csv'
df_ucsf = pd.read_csv(file_path2)
df_ucsf.columns = ['Unnamed', 'age_at_meas', 'measurement', 'mean_val', 'stdev']
df_ucsf['age_at_meas'] = df_ucsf['age_at_meas'] / 7  # Convert to weeks

cell_types = ['Neutrophils', 'Eosinophils', 'Lymphocytes', 'Monocytes', 'Basophils']
intervals = [(1, 7), (7, 13), (13, 19), (19, 25)]

# Function to calculate slopes per interval
def calculate_interval_slopes(df, x_col, y_col, intervals):
    slopes = []
    for start, end in intervals:
        subset = df[(df[x_col] >= start) & (df[x_col] <= end)]
        if len(subset) > 1:
            slope, _, _, _, _ = linregress(subset[x_col], subset[y_col])
            slopes.append(slope)
        else:
            slopes.append(np.nan)
    return slopes

# Collect slopes for Stanford and UCSF
results = []
for cell_type in cell_types:
    col_name = f'{cell_type}/100 leukocytes in Blood'
    if col_name in df_stanford.columns:
        stanford_df = df_stanford[['age_at_meas', col_name]].dropna()
        ucsf_df = df_ucsf[df_ucsf['measurement'].str.lower() == cell_type.lower()]

        stanford_slopes = calculate_interval_slopes(stanford_df, 'age_at_meas', col_name, intervals)
        ucsf_slopes = calculate_interval_slopes(ucsf_df, 'age_at_meas', 'mean_val', intervals)

        # Compute correlation
        valid_idx = ~np.isnan(stanford_slopes) & ~np.isnan(ucsf_slopes)
        if valid_idx.sum() > 1:
            corr, p_corr = pearsonr(np.array(stanford_slopes)[valid_idx], np.array(ucsf_slopes)[valid_idx])
            try:
                _, p_test = wilcoxon(np.array(stanford_slopes)[valid_idx], np.array(ucsf_slopes)[valid_idx])
            except ValueError:
                p_test = np.nan
        else:
            corr, p_corr, p_test = np.nan, np.nan, np.nan

        results.append({
            'Cell Type': cell_type,
            'Correlation': corr,
            'Corr p-value': p_corr,
            'Wilcoxon p-value': p_test,
            **{f'Stanford Slope {start}-{end}w': stanford_slopes[i] for i, (start, end) in enumerate(intervals)},
            **{f'UCSF Slope {start}-{end}w': ucsf_slopes[i] for i, (start, end) in enumerate(intervals)}
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Print results in table format
print("\n=== Trend Similarity Results (Stanford vs UCSF) ===\n")
print(results_df.to_string(index=False))

# Heatmap for correlation (diagonal only since correlation per cell type)
plt.figure(figsize=(8, 5))
sns.heatmap(results_df[['Correlation']].set_index(results_df['Cell Type']), annot=True, cmap='coolwarm', center=0, cbar_kws={'label': 'Correlation'})
plt.title('Stanford vs UCSF Trend Correlation (by Cell Type)')
plt.ylabel('Cell Type')
plt.show()


# In[19]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, norm
import warnings

warnings.filterwarnings("ignore")

# Load data
file_path1 = "/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv"
df_stanford = pd.read_csv(file_path1)

file_path2 = '/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/neelu_validation_stuffs/day_summary_for_stanford.csv'
df_ucsf = pd.read_csv(file_path2)
df_ucsf.columns = ['Unnamed', 'age_at_meas', 'measurement', 'mean_val', 'stdev']
df_ucsf['age_at_meas'] = df_ucsf['age_at_meas'] / 7  # Convert to weeks

cell_types = ['Neutrophils', 'Eosinophils', 'Lymphocytes', 'Monocytes', 'Basophils']
intervals = [(1, 7), (7, 13), (13, 19), (19, 25)]

# Function to calculate slope and SE per interval
def calculate_interval_slopes(df, x_col, y_col, intervals):
    slopes = []
    ses = []
    for start, end in intervals:
        subset = df[(df[x_col] >= start) & (df[x_col] <= end)]
        if len(subset) > 1:
            slope, intercept, r, p, se = linregress(subset[x_col], subset[y_col])
            slopes.append(slope)
            ses.append(se)
        else:
            slopes.append(np.nan)
            ses.append(np.nan)
    return slopes, ses

# Collect results
results = []
for cell_type in cell_types:
    col_name = f'{cell_type}/100 leukocytes in Blood'
    if col_name in df_stanford.columns:
        stanford_df = df_stanford[['age_at_meas', col_name]].dropna()
        ucsf_df = df_ucsf[df_ucsf['measurement'].str.lower() == cell_type.lower()]

        stanford_slopes, stanford_ses = calculate_interval_slopes(stanford_df, 'age_at_meas', col_name, intervals)
        ucsf_slopes, ucsf_ses = calculate_interval_slopes(ucsf_df, 'age_at_meas', 'mean_val', intervals)

        # Compute correlation of slopes across intervals
        valid_idx = ~np.isnan(stanford_slopes) & ~np.isnan(ucsf_slopes)
        if valid_idx.sum() > 1:
            corr, p_corr = pearsonr(np.array(stanford_slopes)[valid_idx], np.array(ucsf_slopes)[valid_idx])
        else:
            corr, p_corr = np.nan, np.nan

        # Compare slopes per interval using t-test formula
        ttest_results = []
        for i in range(len(intervals)):
            if not np.isnan(stanford_slopes[i]) and not np.isnan(ucsf_slopes[i]):
                diff = stanford_slopes[i] - ucsf_slopes[i]
                se_combined = np.sqrt(stanford_ses[i]**2 + ucsf_ses[i]**2)
                t_stat = diff / se_combined if se_combined > 0 else np.nan
                p_val = 2 * norm.sf(abs(t_stat)) if not np.isnan(t_stat) else np.nan
            else:
                t_stat, p_val = np.nan, np.nan
            ttest_results.append((t_stat, p_val))

        results.append({
            'Cell Type': cell_type,
            'Correlation': corr,
            'Corr p-value': p_corr,
            **{f'Stanford Slope {start}-{end}w': stanford_slopes[i] for i, (start, end) in enumerate(intervals)},
            **{f'UCSF Slope {start}-{end}w': ucsf_slopes[i] for i, (start, end) in enumerate(intervals)},
            **{f'p-value {start}-{end}w': ttest_results[i][1] for i, (start, end) in enumerate(intervals)}
        })

results_df = pd.DataFrame(results)

# Print full table
print("\n=== Stanford vs UCSF Slope Comparison with t-test (per interval) ===\n")
print(results_df.to_string(index=False))

# Heatmap for correlation across cell types
plt.figure(figsize=(8, 5))
sns.heatmap(results_df[['Correlation']].set_index(results_df['Cell Type']), annot=True, cmap='coolwarm', center=0)
plt.title('Stanford vs UCSF Trend Correlation (by Cell Type)')
plt.ylabel('Cell Type')
plt.show()


# In[20]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

# Load the data
file_path1 = "/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv"
df_measures_units = pd.read_csv(file_path1)

# Cell types to compare
cell_types = ['Neutrophils', 'Eosinophils', 'Lymphocytes', 'Monocytes', 'Basophils']

# Define intervals
intervals = [(1, 7), (7, 13), (13, 19), (19, 25)]

# Function to calculate mean values for intervals
def calculate_interval_means(df, col_name, intervals):
    means = []
    for start, end in intervals:
        interval = df[(df['age_at_meas'] >= start) & (df['age_at_meas'] <= end)]
        if not interval.empty:
            means.append(interval[col_name].mean())
        else:
            means.append(np.nan)
    return means

# Function to plot regression lines and annotate slope
def annotate_slope(ax, x, y, start, end):
    if len(x) > 1 and len(y) > 1:
        slope, intercept, _, _, _ = linregress(x, y)
        midpoint = (start + end) / 2
        y_pos = ax.get_ylim()[0] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.annotate(f'Slope: {slope:.2f}', xy=(midpoint, y_pos), color='black', ha='center')
        return slope
    else:
        print(f"Insufficient data points to calculate slope: {len(x)} points")
        return None

# Define colors for each cell type
colors = {
    'Basophils': '#ecb6eb',
    'Neutrophils': 'orange',
    'Eosinophils': '#ecb6eb',
    'Lymphocytes': '#ecb6eb',
    'Monocytes': 'red'
}

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
fig.suptitle('B. White blood cell counts over time.', fontsize=16, weight='bold')

for i, cell_type in enumerate(cell_types):
    col_name = f'{cell_type}/100 leukocytes in Blood'

    # Plot Stanford data
    if col_name in df_measures_units.columns:
        test = df_measures_units[['age_at_meas', col_name]].dropna()

        # Calculate mean values for intervals
        interval_means = calculate_interval_means(test, col_name, intervals)
        interval_midpoints = [(start + end) / 2 for start, end in intervals]

        # Create bar plot for interval means
        axes[i].bar(interval_midpoints, interval_means, width=4, color='lightblue', alpha=0.5)

        # Add trend line connecting the midpoints
        axes[i].plot(interval_midpoints, interval_means, color='gray', marker='o', linestyle='-', linewidth=2, markersize=6)

        # Overlay line plot
        sns.lineplot(data=test, x='age_at_meas', y=col_name, ax=axes[i], ci=None, color=colors[cell_type])

        axes[i].set_title(f'{cell_type}/100 leukocytes in Blood')
        axes[i].set_xlabel('Age at measurement (weeks)')
        axes[i].set_ylabel('Value of the measurement')
        axes[i].tick_params(axis='x')

        # Annotate slopes for Stanford data
        for j, (start, end) in enumerate(intervals):
            interval = test[(test['age_at_meas'] >= start) & (test['age_at_meas'] <= end)]
            if not interval.empty:
                print(f"Processing {cell_type} for Stanford: Interval {start}-{end} weeks, {len(interval)} points")
                print(interval[['age_at_meas', col_name]])  # Debugging: Check interval data
                slope = annotate_slope(axes[i], interval['age_at_meas'], interval[col_name], start, end)
                if slope is not None:
                    axes[i].axvline(x=start, linestyle='--', color='gray', alpha=0.5)
                    axes[i].axvline(x=end, linestyle='--', color='gray', alpha=0.5)
                    print(f'{cell_type} - Stanford Slope ({start}-{end} weeks): {slope}')
            else:
                print(f"No data for {cell_type} in Stanford: Interval {start}-{end} weeks")

# Remove any unused subplot axes
for j in range(len(cell_types), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('stanford_cohort_cbc_comparison_colored_July23.png', dpi=300)
plt.show()


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

# Load the data
file_path1 = "/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv"
df_measures_units = pd.read_csv(file_path1)

# Cell types to compare
cell_types = ['Neutrophils', 'Eosinophils', 'Lymphocytes', 'Monocytes', 'Basophils']

# Define intervals
intervals = [(1, 7), (7, 13), (13, 19), (19, 25)]

# Function to calculate mean values for intervals
def calculate_interval_means(df, col_name, intervals):
    means = []
    for start, end in intervals:
        interval = df[(df['age_at_meas'] >= start) & (df['age_at_meas'] <= end)]
        if not interval.empty:
            means.append(interval[col_name].mean())
        else:
            means.append(np.nan)
    return means

# Function to plot regression lines and annotate slope
def annotate_slope(ax, x, y, start, end):
    if len(x) > 1 and len(y) > 1:
        slope, intercept, _, _, _ = linregress(x, y)
        midpoint = (start + end) / 2
        y_pos = ax.get_ylim()[0] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.annotate(f'Slope: {slope:.2f}', xy=(midpoint, y_pos), color='black', ha='center')
        return slope
    else:
        print(f"Insufficient data points to calculate slope: {len(x)} points")
        return None

# Define colors for each cell type
colors = {
    'Basophils': '#e80708',     # Red
    'Neutrophils': '#6295ee',   # Blue
    'Eosinophils': '#ed8d18',   # Yellow
    'Lymphocytes': '#ed8d18',   # Yellow
    'Monocytes': '#cf75ca'      # Purple
}

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
fig.suptitle('B. White blood cell counts over time.', fontsize=16, weight='bold')

for i, cell_type in enumerate(cell_types):
    col_name = f'{cell_type}/100 leukocytes in Blood'

    # Plot Stanford data
    if (col_name in df_measures_units.columns):
        test = df_measures_units[['age_at_meas', col_name]].dropna()

        # Calculate mean values for intervals
        interval_means = calculate_interval_means(test, col_name, intervals)
        interval_midpoints = [(start + end) / 2 for start, end in intervals]

        # Create bar plot for interval means
        axes[i].bar(interval_midpoints, interval_means, width=4, color='lightblue', alpha=0.5)

        # Add trend line connecting the midpoints
        axes[i].plot(interval_midpoints, interval_means, color='gray', marker='o', linestyle='-', linewidth=2, markersize=6)

        # Overlay line plot
        sns.lineplot(data=test, x='age_at_meas', y=col_name, ax=axes[i], ci=None, color=colors[cell_type])

        axes[i].set_title(f'{cell_type}/100 leukocytes in Blood')
        axes[i].set_xlabel('Age at measurement (weeks)')
        axes[i].set_ylabel('Value of the measurement')
        axes[i].tick_params(axis='x')

        # Annotate slopes for Stanford data
        for j, (start, end) in enumerate(intervals):
            interval = test[(test['age_at_meas'] >= start) & (test['age_at_meas'] <= end)]
            if not interval.empty:
                print(f"Processing {cell_type} for Stanford: Interval {start}-{end} weeks, {len(interval)} points")
                print(interval[['age_at_meas', col_name]])  # Debugging: Check interval data
                slope = annotate_slope(axes[i], interval['age_at_meas'], interval[col_name], start, end)
                if slope is not None:
                    axes[i].axvline(x=start, linestyle='--', color='gray', alpha=0.5)
                    axes[i].axvline(x=end, linestyle='--', color='gray', alpha=0.5)
                    print(f'{cell_type} - Stanford Slope ({start}-{end} weeks): {slope}')
            else:
                print(f"No data for {cell_type} in Stanford: Interval {start}-{end} weeks")

# Remove any unused subplot axes
for j in range(len(cell_types), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('stanford_cohort_cbc_comparison_colored_July23.png', dpi=300)
plt.show()


# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

# Load the data
file_path1 = "/Users/sshome/Desktop/backup_Jan3_2025/Bioinformatics_projects/Neelu/Neelu_project/Neonatal_measurements_final_share_final_final_check_this_only/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv"
df_measures_units = pd.read_csv(file_path1)

# Cell types to compare
cell_types = ['Neutrophils', 'Eosinophils', 'Lymphocytes', 'Monocytes', 'Basophils']

# Define intervals
intervals = [(1, 7), (7, 13), (13, 19), (19, 25)]

# Function to calculate mean values for intervals
def calculate_interval_means(df, col_name, intervals):
    means = []
    for start, end in intervals:
        interval = df[(df['age_at_meas'] >= start) & (df['age_at_meas'] <= end)]
        if not interval.empty:
            means.append(interval[col_name].mean())
        else:
            means.append(np.nan)
    return means

# Function to plot regression lines and annotate slope
def annotate_slope(ax, x, y, start, end):
    if len(x) > 1 and len(y) > 1:
        slope, intercept, _, _, _ = linregress(x, y)
        midpoint = (start + end) / 2
        y_pos = ax.get_ylim()[0] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.annotate(f'{slope:.2f}', xy=(midpoint, y_pos), color='black',
                    ha='center', fontsize=12, fontweight='bold')
        return slope
    else:
        print(f"Insufficient data points to calculate slope: {len(x)} points")
        return None

# Set global font size for better readability
sns.set_context("talk", font_scale=1.4)  # Options: "paper", "notebook", "talk", "poster"

# Define colors for each cell type
colors = {
    'Basophils': '#e63946',     # Red
    'Neutrophils': '#6495ED',   # Blue
    'Eosinophils': '#ef8c01',   # Orange
    'Lymphocytes': '#ef8c01',   # Purple
    'Monocytes': '#DA70D6'      #teal
}

#009688
# Teal (adjusted for better distinction)

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()
fig.suptitle('B. White blood cell counts over time.', fontsize=22, weight='bold')

for i, cell_type in enumerate(cell_types):
    col_name = f'{cell_type}/100 leukocytes in Blood'

    if (col_name in df_measures_units.columns):
        test = df_measures_units[['age_at_meas', col_name]].dropna()

        # Calculate mean values for intervals
        interval_means = calculate_interval_means(test, col_name, intervals)
        interval_midpoints = [(start + end) / 2 for start, end in intervals]

        # Bar plot for interval means
        axes[i].bar(interval_midpoints, interval_means, width=4, color='lightblue', alpha=0.5)

        # Add trend line connecting the midpoints
        axes[i].plot(interval_midpoints, interval_means, color='gray', marker='o', linestyle='-', linewidth=2, markersize=8)

        # Overlay line plot
        sns.lineplot(data=test, x='age_at_meas', y=col_name, ax=axes[i], ci=None, color=colors[cell_type])

        axes[i].set_title(f'{cell_type}/100 leukocytes in Blood', fontsize=18, fontweight='bold')
        axes[i].set_xlabel('Age at measurement (weeks)', fontsize=16)
        axes[i].set_ylabel('Value of the measurement', fontsize=16)
        axes[i].tick_params(axis='both', which='major', labelsize=14)

        # Annotate slopes for Stanford data
        for j, (start, end) in enumerate(intervals):
            interval = test[(test['age_at_meas'] >= start) & (test['age_at_meas'] <= end)]
            if not interval.empty:
                slope = annotate_slope(axes[i], interval['age_at_meas'], interval[col_name], start, end)
                if slope is not None:
                    axes[i].axvline(x=start, linestyle='--', color='gray', alpha=0.5)
                    axes[i].axvline(x=end, linestyle='--', color='gray', alpha=0.5)

# Remove unused subplot axis
for j in range(len(cell_types), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('stanford_cohort_cbc_comparison_colored_July31_larger_fonts.png', dpi=300)
plt.show()


# In[ ]:




