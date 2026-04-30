#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pymannkendall')
get_ipython().system('pip install pyreadr')
get_ipython().system('pip install statsmod')
get_ipython().system('pip install ipdb')


# In[2]:


get_ipython().run_line_magic('reset', '')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pymannkendall as mk
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.stats
import sklearn.manifold
from time import time
import os
from scipy.interpolate import CubicSpline,UnivariateSpline,splprep, splev
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyreadr 
from statsmodels.sandbox.stats.multicomp import multipletests
from math import sqrt
import random
from cycler import cycler
import pickle 
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import ipdb

seed = 49



random.seed(seed)
np.random.seed(seed)


# In[3]:


import sklearn
import pymannkendall
import pandas
import scipy
import statsmodels
import networkx
import pyreadr
print(f"scikit-learn version: {sklearn.__version__}")
print(f"pymannkendall version: {pymannkendall.__version__}")
print(f"pandas version: {pandas.__version__}")
print(f"scipy version: {scipy.__version__}")
print(f"statsmodels version: {statsmodels.__version__}")
print(f"networkx version: {networkx.__version__}")
print(f"pyreadr version: {pyreadr.__version__}")


# In[ ]:





# In[5]:


rmv_vars = ["person_id",'measurement_DATE','birth_DATETIME','age_at_meas']      


# In[6]:


df_measurements_6mnt_outl_rmv = pd.read_csv("../Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv")
df_measurements_6mnt_outl_rmv.columns = df_measurements_6mnt_outl_rmv.apply(lambda x: ''.join(filter(str.isalnum, str(x.name))))
df_measurements_6mnt_outl_rmv = df_measurements_6mnt_outl_rmv.apply(lambda x: x.fillna(x.mean()),axis=0)


# In[ ]:


## trends significance 
p_values = pd.read_csv("../Data/linear_mixed_eff_trends_WBCagg_measuresScaled.csv")
p_values = p_values[~(p_values['p_value']=='error')]
p_values = p_values[~(p_values['p_value']== np.nan)]

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
p_values['Variables'].replace('X17Hydroxypregnenolonemeasurement','17Hydroxypregnenolonemeasurement',inplace=True)


p_values = p_values[p_values['Variables'].isin(df_measurements_6mnt_outl_rmv.columns)]



method = 'fdr_bh' #fdr_bh, bonferroni
p_values_adj = p_values

p_values_adj['adj_p_values'] = multipletests(p_values['p_value'].to_list(), method=method, returnsorted=False)[1]
p_values_adj['include'] = multipletests(p_values['p_value'].to_list(), method=method, returnsorted=False)[0]

p_values_adj['adj_log_10'] = -np.log10(p_values_adj['adj_p_values'])
max_value = np.nanmax(p_values_adj['adj_log_10'][p_values_adj['adj_log_10'] != np.inf])

#replace inf and -inf in column with max value of column 
p_values_adj['adj_log_10'].replace([np.inf, -np.inf], max_value, inplace=True)
p_values_adj = p_values_adj[p_values_adj['include']]

## save the results 
# p_values_adj.to_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/trends_significance_BHadjusted.csv", index=False)
df_measurements_6mnt_outl_rmv = df_measurements_6mnt_outl_rmv.drop(columns=[col for col in df_measurements_6mnt_outl_rmv if col not in p_values_adj['Variables'].to_list()])
df = pd.pivot_table(p_values, values ='adj_log_10',columns=['Variables'])



# In[ ]:


## GA effect on significant trends 

p_values_GA = pd.read_csv("../Data/linear_mixed_eff_GA.csv")
p_values_GA['p_value'] = p_values_GA['p_value'].replace('error',1)
p_values_GA['p_value'] = p_values_GA['p_value'].replace(np.nan,1)

p_values_GA['p_value'] = pd.to_numeric(p_values_GA['p_value'])
p_values_GA['log_10_p_value'] = -np.log10(p_values_GA['p_value'])
max_value = np.nanmax(p_values_GA['log_10_p_value'][p_values_GA['log_10_p_value'] != np.inf])

#replace inf and -inf in column with max value of column 
p_values_GA['log_10_p_value'].replace([np.inf, -np.inf], max_value, inplace=True) 
p_values_GA.Variables = p_values_GA.Variables.str.replace('[#,@,&,.,/]', '')
p_values_GA['Variables'].replace('X25hydroxyvitaminD3MassvolumeinSerumorPlasma', '25hydroxyvitaminD3MassvolumeinSerumorPlasma',inplace=True)
p_values_GA['Variables'].replace('X11DeoxycortisolMassvolumeinSerumorPlasma', '11DeoxycortisolMassvolumeinSerumorPlasma',inplace=True)
p_values_GA['Variables'].replace('X17Hydroxyprogesteronemeasurement', '17Hydroxyprogesteronemeasurement',inplace=True)
p_values_GA['Variables'].replace('X2MethylbutyrylglycineCreatinineMassRatioinUrine', '2MethylbutyrylglycineCreatinineMassRatioinUrine',inplace=True)
p_values_GA['Variables'].replace('X25HydroxyvitaminD325HydroxyvitaminD2MassvolumeinSerumorPlasma', '25HydroxyvitaminD325HydroxyvitaminD2MassvolumeinSerumorPlasma',inplace=True)
p_values_GA['Variables'].replace('X25hydroxyvitaminD3MassvolumeinSerumorPlasma', '25hydroxyvitaminD3MassvolumeinSerumorPlasma',inplace=True)
p_values_GA['Variables'].replace('X13betaglucanMassvolumeinSerum','13betaglucanMassvolumeinSerum',inplace=True)
p_values_GA['Variables'].replace('X10HydroxycarbazepineMassvolumeinSerumorPlasma','10HydroxycarbazepineMassvolumeinSerumorPlasma',inplace=True)
p_values_GA['Variables'].replace('X17Hydroxypregnenolonemeasurement','17Hydroxypregnenolonemeasurement',inplace=True)

p_values_GA = p_values_GA[p_values_GA['Variables'].isin(df_measurements_6mnt_outl_rmv.columns)]


method = 'fdr_bh' #fdr_bh, bonferroni
p_values_GA_adj = p_values_GA

p_values_GA_adj['adj_p_values'] = multipletests(p_values_GA['p_value'].to_list(), method=method, returnsorted=False)[1]
p_values_GA_adj['include'] = multipletests(p_values_GA['p_value'].to_list(), method=method, returnsorted=False)[0]

p_values_GA_adj['adj_log_10'] = -np.log10(p_values_GA_adj['adj_p_values'])
max_value = np.nanmax(p_values_GA_adj['adj_log_10'][p_values_GA_adj['adj_log_10'] != np.inf])

#replace inf and -inf in column with max value of column 
p_values_GA_adj['adj_log_10'].replace([np.inf, -np.inf], max_value, inplace=True)

## Only consider the node size for the significant trends  

p_values_GA_adj = p_values_GA_adj[p_values_GA_adj['Variables'].isin(p_values_adj['Variables'])]
p_values_GA_adj.to_csv("../Data/GA_significance_BHadjusted.csv", index=False)


df = pd.pivot_table(p_values_GA_adj, values ='adj_log_10',columns=['Variables'])



# In[ ]:


## Congenital anomalies of circ effect on significant trends 

p_values_CA = pd.read_csv("../Data/linear_mixed_eff_trends_updatedData_measuresScaled_effectofCongCircDisease.csv")
p_values_CA['p_value'] = p_values_CA['p_value'].replace('error',1)
p_values_CA['p_value'] = p_values_CA['p_value'].replace(np.nan,1)

p_values_CA['p_value'] = pd.to_numeric(p_values_CA['p_value'])
p_values_CA['log_10_p_value'] = -np.log10(p_values_CA['p_value'])
max_value = np.nanmax(p_values_CA['log_10_p_value'][p_values_CA['log_10_p_value'] != np.inf])

#replace inf and -inf in column with max value of column 
p_values_CA['log_10_p_value'].replace([np.inf, -np.inf], max_value, inplace=True)
p_values_CA.Variables = p_values_CA.Variables.str.replace('[#,@,&,.,/]', '')
p_values_CA['Variables'].replace('X25hydroxyvitaminD3MassvolumeinSerumorPlasma', '25hydroxyvitaminD3MassvolumeinSerumorPlasma',inplace=True)
p_values_CA['Variables'].replace('X11DeoxycortisolMassvolumeinSerumorPlasma', '11DeoxycortisolMassvolumeinSerumorPlasma',inplace=True)
p_values_CA['Variables'].replace('X17Hydroxyprogesteronemeasurement', '17Hydroxyprogesteronemeasurement',inplace=True)
p_values_CA['Variables'].replace('X2MethylbutyrylglycineCreatinineMassRatioinUrine', '2MethylbutyrylglycineCreatinineMassRatioinUrine',inplace=True)
p_values_CA['Variables'].replace('X25HydroxyvitaminD325HydroxyvitaminD2MassvolumeinSerumorPlasma', '25HydroxyvitaminD325HydroxyvitaminD2MassvolumeinSerumorPlasma',inplace=True)
p_values_CA['Variables'].replace('X25hydroxyvitaminD3MassvolumeinSerumorPlasma', '25hydroxyvitaminD3MassvolumeinSerumorPlasma',inplace=True)
p_values_CA['Variables'].replace('X13betaglucanMassvolumeinSerum','13betaglucanMassvolumeinSerum',inplace=True)
p_values_CA['Variables'].replace('X10HydroxycarbazepineMassvolumeinSerumorPlasma','10HydroxycarbazepineMassvolumeinSerumorPlasma',inplace=True)
p_values_CA['Variables'].replace('X17Hydroxypregnenolonemeasurement','17Hydroxypregnenolonemeasurement',inplace=True)

p_values_CA = p_values_CA[p_values_CA['Variables'].isin(df_measurements_6mnt_outl_rmv.columns)]


method = 'fdr_bh' #fdr_bh, bonferroni
p_values_CA_adj = p_values_CA

p_values_CA_adj['adj_p_values'] = multipletests(p_values_CA['p_value'].to_list(), method=method, returnsorted=False)[1]
p_values_CA_adj['include'] = multipletests(p_values_CA['p_value'].to_list(), method=method, returnsorted=False)[0]

p_values_CA_adj['adj_log_10'] = -np.log10(p_values_CA_adj['adj_p_values'])
max_value = np.nanmax(p_values_CA_adj['adj_log_10'][p_values_CA_adj['adj_log_10'] != np.inf])

#replace inf and -inf in column with max value of column 
p_values_CA_adj['adj_log_10'].replace([np.inf, -np.inf], max_value, inplace=True)

## Only consider the node size for the significant trends  

p_values_CA_adj = p_values_CA_adj[p_values_CA_adj['Variables'].isin(p_values_adj['Variables'])]
p_values_CA_adj.to_csv("../Data/CongAnom_significance_BHadjusted.csv", index=False)


df = pd.pivot_table(p_values_CA_adj, values ='adj_log_10',columns=['Variables'])



# In[ ]:


# df_measurements_6mnt_outl_rmv = pd.read_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv").\
# drop(columns=rmv_vars)
# df_measurements_6mnt_outl_rmv.columns = df_measurements_6mnt_outl_rmv.apply(lambda x: ''.join(filter(str.isalnum, str(x.name))))
# df_measurements_6mnt_outl_rmv = df_measurements_6mnt_outl_rmv.apply(lambda x: x.fillna(x.mean()),axis=0)

# p_values = pd.read_csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/trends_significance_BHadjusted.csv")
# p_values_inc_trends = p_values[p_values['Include']]

# df_measurements_6mnt_outl_rmv = df_measurements_6mnt_outl_rmv.drop(columns=[col for col in df_measurements_6mnt_outl_rmv if col not in p_values_inc['Variables'].to_list()])

# df = pd.pivot_table(p_values_inc, values ='adj_log_10',columns=['Variables'])


# In[ ]:


import os
import sys
import warnings


def set_threads_for_external_libraries(n_threads=1):
    """
    Tries to disable BLAS or similar implicit  parallelization engines
    by setting the following environment attributes to `1`:
    - OMP_NUM_THREADS
    - OPENBLAS_NUM_THREADS
    - MKL_NUM_THREADS
    - VECLIB_MAXIMUM_THREADS
    - NUMEXPR_NUM_THREADS
    This can be useful since `numpy` and co. are using these libraries to parallelize vector and matrix operations.
    However, by default, the grab ALL CPUs an a machine. Now, if you use parallelization, e.g., based on `joblib` as in
    `sklearn.model_selection.GridSearchCV` or `sklearn.model_selection.cross_validate`, then you will overload your
    machine and the OS scheduler will spend most of it's time switching contexts instead of calculating.
    Parameters
    ----------
    n_threads: int, optional, default: 1
        Number of threads to use for
    Notes
    -----
    - This ONLY works if you import this file BEFORE `numpy` or similar libraries.
    - BLAS and co. only kick in when the data (matrices etc.) are sufficiently large.
      So you might not always see the `CPU stealing` behavior.
    - For additional info see: "2.4.7 Avoiding over-subscription of CPU ressources" in the
      `joblib` docs (https://buildmedia.readthedocs.org/media/pdf/joblib/latest/joblib.pdf).
    - Also note that there is a bug report for `joblib` not disabling BLAS and co.
      appropriately: https://github.com/joblib/joblib/issues/834
    Returns
    -------
    None
    """

    if (
            "numpy" in sys.modules.keys()
            or "scipy" in sys.modules.keys()
            or "sklearn" in sys.modules.keys()
            ):
        warnings.warn("This function should be called before `numpy` or similar modules are imported.")

    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)


# In[ ]:


# reduce threads otherwise the TSNE can take forever
# NOTE: Needs to be called before import `numpy`
n_threads = 1
set_threads_for_external_libraries(n_threads)


# In[ ]:


import numpy as np
import networkx as nx

import matplotlib.pyplot as plt


def nx_plot(
        graph=None,
        nodes=None,
        nodes_pos=None,
        nodes_args=None,
        nodes_labels=None,
        nodes_labels_pos=None,
        nodes_labels_args=None,
        edges=None,
        edges_pos=None,
        edges_args=None,
        edges_labels=None,
        edges_labels_pos=None,
        edges_labels_args=None,
        ax=None):
    """More complete version of`networkx` plotting. 
    For formatting please refer to:
    * `nx.draw_networkx_nodes`
    * `nx.draw_networkx_edges`
    * `nx.draw_networkx_edge_labels`
    * `matplotlib.pyplot.annotate`
    TODO: 
        * document arguments
        * add examples
    Notes:
    * Alpha for individual edges: set alpha in color using `edge_color`
    * NetworkX = 2.6.3
        * Self-loops are plotted all of a sudden but their formatting doesn't work as expected. 
          Luckily this does not influence non-self-loops:
          Link: https://github.com/networkx/networkx/issues/5106
    * NetworkX < 2.4:
        * Weirdness to make **edges transparent**:
            Set edge alpha to a array or a function (value does not matter),
            and then the set edge colors to RGBA:
            `edges_args=dict(edge_color=lambda g,src,dst,d: return (1,0,0,d['alpha']), alpha=lambda g,src,dst,d: 1)`
        * When using matplotlib.pyplot.subplots networkx adjusted axes by calling `plt.tick_params` causing unwanted
            behavior. This was fixed in networkx 2.4.
    Parameters
    ----------
    graph : [type]
        [description]
    nodes : [type]
        [description]
    nodes_pos : [type]
        [description]
    nodes_args : [type], optional
        [description] (the default is None, which [default_description])
    nodes_labels : [type], optional
        [description] (the default is None, which [default_description])
    nodes_labels_pos : [type], optional
        [description] (the default is None, which [default_description])
    nodes_labels_args : [type], optional
        [description] (the default is None, which [default_description])
    edges : [type], optional
        [description] (the default is None, which [default_description])
    edges_pos : [type], optional
        [description] (the default is None, which [default_description])
    edges_args : [type], optional
        [description] (the default is None, which [default_description])
    edges_labels : [type], optional
        [description] (the default is None, which [default_description])
    edges_labels_pos : [type], optional
        [description] (the default is None, which [default_description])
    edges_labels_args : [type], optional
        [description] (the default is None, which [default_description])

    Returns
    -------
    [type]
        [description]
    """
    #set color map

#     plt.set_cmap('Set3')
#     plt.set_cmap('Set2_r')




#     plt.grid(False)
#     plt.rcParams['figure.facecolor'] = 'white'
    plt.box(False)




    # get axis
    if ax is None:
        ax = plt.gca()
#         ax.set_facecolor('white')

    # init graph

    if graph is None or isinstance(graph, str):
        if graph == "bi" or graph == "di":
            g = nx.DiGraph()
        else:
            g = nx.Graph()
    else:
        g = graph

    # init nodes

    if graph is None and nodes is None:
        if nodes_pos is not None:
            if isinstance(nodes_pos, dict):
                nodes = nodes_pos.keys()
            else: 
                nodes = np.arange(nodes_pos.shape[0])
        else:
            raise Exception("Either `graph` or `nodes` must be given. Both were `None`.")

    if nodes is not None:
        if isinstance(nodes, dict):
            nodes = [(k, v) for k, v in nodes.items()]
        g.add_nodes_from(nodes)

    # init edges

    if edges is not None:
        if isinstance(edges, dict):
            edges = [(*k, v) for k, v in edges.items()]
        g.add_edges_from(edges)

    # init positions

    def init_pos(pos):
        if pos is None:
            return nodes_pos
        elif callable(pos):
            return {n: pos(g, n, d) for n, d in g.nodes(data=True)}
        elif isinstance(pos, dict):
            return pos
        else:
            return {n: p for n, p in zip(g.nodes(), pos)}

    nodes_pos = init_pos(nodes_pos)
    nodes_labels_pos = init_pos(nodes_labels_pos)
    edges_pos = init_pos(edges_pos)
    edges_labels_pos = init_pos(edges_labels_pos)

    # init labels

    def init_nodes_labels(labels):
        if callable(labels):
            return {n: labels(g, n, d) for n, d in g.nodes(data=True)}
        else:
            return labels
    nodes_labels = init_nodes_labels(nodes_labels)

    def init_edges_labels(labels):
        if callable(labels):
            tmp = {(src, dst): labels(g, src, dst, d) for src, dst, d in g.edges(data=True)}
            tmp = {k: v for k, v in tmp.items() if v is not None}  # filter "None" labels
            return tmp
        else:
            return labels
    edges_labels = init_edges_labels(edges_labels)

    # init layout arguments

    def init_node_args(args):
        if args is None:
            args = {}
        else:
            args = args.copy()
            for k, v in args.items():
                if callable(v):
                    args[k] = [v(g, n, d) for n, d in g.nodes(data=True)]
        if "ax" not in args:
            args["ax"] = ax
        return args

    nodes_args = init_node_args(nodes_args)
    nodes_labels_args = init_node_args(nodes_labels_args)

    def init_edges_args(args):
        if args is None:
            args = {}
        else:
            args = args.copy()
            for k, v in args.items():
                if callable(v):
                    args[k] = [v(g, src, dst, d) for src, dst, d in g.edges(data=True)]
        if "ax" not in args:
            args["ax"] = ax
        return args

    edges_args = init_edges_args(edges_args)
    edges_labels_args = init_edges_args(edges_labels_args)

    # draw nodes (allow for several of shapes for nodes)
    if "node_shape" in nodes_args and type(nodes_args["node_shape"]) is list:

        shapes = list(zip(range(len(g.nodes())), nodes_args["node_shape"]))
        unique_shapes = np.unique(nodes_args["node_shape"])

        for shape in unique_shapes:

            shape_idx = [i for i, s in shapes if s == shape]

            nodes = list(g.nodes())
            nodelist = [nodes[i] for i in shape_idx]

            shape_args = nodes_args.copy()
            del shape_args["node_shape"]

            for arg, _ in shape_args.items():
                if type(shape_args[arg]) is list:
                    shape_args[arg] = [shape_args[arg][i] for i in shape_idx]

            nx.draw_networkx_nodes(g, nodes_pos, nodelist=nodelist, node_shape=shape, **shape_args)
    else:
        nx.draw_networkx_nodes(g, nodes_pos, **nodes_args)

    # draw edges
    # print(g.nodes(data=True))
    # print(g.edges(data=True))
    # print(edges_args)
    nx.draw_networkx_edges(g, nodes_pos, **edges_args)

    # draw node labels
    if nodes_labels is not None:

        # rename args for compatibility to `nx.draw_networkx_labels`
        args = nodes_labels_args.copy()
        del args["ax"]
        for original_key, new_key in {
                "font_size": "fontsize",
                "font_color": "color",
                "font_family": "family",
                "font_weight": "weight"}.items():
            if original_key in args:
                args[new_key] = args[original_key]
                del args[original_key]

        # check if we have list args
        list_args = []
        for arg, value in list(args.items()):
            if isinstance(value, list) and len(value) == len(g.nodes):
                list_args.append((arg, value))
                del args[arg]

        for i, node in enumerate(g.nodes):
            ax.annotate(nodes_labels[node], nodes_labels_pos[node], **args, **{a: v[i] for a, v in list_args})

    # draw edge labels
    if edges_labels is not None:
        nx.draw_networkx_edge_labels(g, pos=edges_labels_pos, edge_labels=edges_labels, **edges_labels_args)

    #legend
    for n in [10, 50, 150,200]:
        plt.plot([], [],'o', markersize = sqrt(n), label = f"{n}", color='black',marker='o',markerfacecolor='white')
        plt.legend(labelspacing = 1,loc='upper right',frameon = False, title='$- log_{10}(p value)$')




#      for n in [8.10, 48.64, 140.69,273.22]:
#         plt.plot([], [], 'o', color="tab:blue", markersize = sqrt(n)*2.2, label = f"{n}")
#         plt.legend(labelspacing = 5, loc='center left', bbox_to_anchor=(0, 0.5),frameon = False)

#     for n in [0.173, 0.406,0.754]:
#         plt.plot([], [], color = [0,0,0,abs(n)],marker = '_', linewidth = n*3, label = f"{n}")
#         plt.legend(labelspacing = 5, loc='center left', bbox_to_anchor=(0, 0.5), frameon = False)

#     plt.tick_params(axis='x', which='both', bottom=False,
#                 top=False, labelbottom=False)

#     # Selecting the axis-Y making the right and left axes False
#     plt.tick_params(axis='y', which='both', right=False,
#                 left=False, labelleft=False)

    return 


# In[ ]:


n_samples = len(X)
n_features = X.shape[1]


# In[ ]:


n_features


import scipy.stats
import numpy as np
import sklearn.manifold

# some random data
# n_samples = 100
# n_features = 20
# X = np.random.random((n_samples, n_features))
# y = np.random.random(n_samples)
# y_labels = np.random.choice([1,2,3], n_features)

#X = df_measurements_6mnt_outl_rmv.drop(columns=['age_at_meas'])
#y = df_measurements_6mnt_outl_rmv['age_at_meas']

X = df_measurements_6mnt_outl_rmv
n_samples = len(X)
n_features = X.shape[1]

#with open("../Data/clustering_labels", "rb") as fp:   # Unpickling
 #   label = pickle.load(fp)

#with open("../Data/clustering_measures", "rb") as fp:   # Unpickling
 #   clustering_measures = pickle.load(fp)
    
#with open("../Data/clustering_measures_names", "rb") as fp:   # Unpickling
 #   clustering_measures_names = pickle.load(fp)


correlation_matrix = np.corrcoef(X, rowvar=False) 

# correlation_matrix = np.array(X.corr())

# from scipy.stats import pearsonr
# def pearsonr_pval(x,y):
#     return pearsonr(x,y)[1]
# correlation_matrix = np.array(X.corr(method=pearsonr_pval))


# correlation_matrix = correlation_matrix.replace(np.nan, 0)
# correlation_matrix = np.array(correlation_matrix)

assert correlation_matrix.shape[0] == n_features

abs_corr_matrix = np.abs(correlation_matrix)

# correlation / association of features to outcome
# y_correlation = np.array([
# #     scipy.stats.spearmanr(y, X.iloc[:,i])
#     scipy.stats.pearsonr(y, X.iloc[:,i])
#     for i in range(n_features)
# ])

perplexity = min(30, n_samples - 1) 

# calculate embedding of features based on absolute values in the correlation matrix
# tsne = sklearn.manifold.TSNE(n_components=2, perplexity = 5, random_state=0, n_iter=10000)
# tsne = sklearn.manifold.TSNE(n_components=2, perplexity = 9, random_state=49)


tsne = sklearn.manifold.TSNE(n_components=2, perplexity=perplexity, random_state=49, n_iter=1000)
#embeddings = tsne.fit_transform(abs_corr_matrix)


#tsne = sklearn.manifold.TSNE(n_components=2, perplexity = 2, random_state=49,n_iter=1000)



# embeddings = tsne.fit_transform(np.abs(correlation_matrix))

embeddings = tsne.fit_transform(label.reshape(-1,1))
# colors_plt = ["#6495ED", "#DA70D6", "#20B2AA", "#FF8C00"]
# colors_plt = ["#6495ED", "#DA70D6", "#20B2AA", "#FF8C00", '#BA70D6','#6615ED']


colors_plt = ["#6495ED", "#DA70D6", "#20B2AA", "#ef8c01", "red"]


# colors_plt = [ (0.39215686274509803, 0.5843137254901961, 0.9294117647058824, 0.7),\
#              (0.8549019607843137, 0.4392156862745098, 0.8392156862745098, 0.7),\
#              (12549019607843137, 0.6980392156862745, 0.6666666666666666, 0.7),\
#              (0.9372549019607843, 0.5490196078431373, 0.00392156862745098, 0.7)]

# colors_plt = ["cornflowerblue", "violet", "lightseagreen", "darkorange"]





# Assuming X is your feature data



# Use absolute values of the correlation matrix for t-SNE


# Ensure the t-SNE perplexity is less than the number of samples
 # 30 is a common default value for perplexity




# In[ ]:


import numpy as np
import sklearn.manifold

X = df_measurements_6mnt_outl_rmv

# Assuming X is your feature data
n_samples = len(X)
n_features = X.shape[1]

correlation_matrix = np.corrcoef(X, rowvar=False)
assert correlation_matrix.shape[0] == n_features

abs_corr_matrix = np.abs(correlation_matrix)

tsne = sklearn.manifold.TSNE(n_components=2, perplexity = 7, random_state=0, n_iter=100000)
embeddings = tsne.fit_transform(abs_corr_matrix)

colors_plt = ["#6495ED", "#DA70D6", "#20B2AA", "#ef8c01", "red"]


# In[ ]:





# In[ ]:


## save the graph
nodes = [
    (i, {
#         "correlation_y": y_correlation[i],
#         "label": y_labels.iloc[i],
        "name": X.iloc[:,i].name,
#         "name":dict_rsq[i][0],
#     "r_sq": dict_rsq[i][1],
#        "r_sq": dict_rsq[i][1].p
#         "r_sq": p_values[i]
#         "p_value": df.iloc[:,i].log_10_p_value
        "p_value": df[X.iloc[:,i].name].adj_log_10,
        "label":colors_plt[label[i]]


    })
    for i in range(n_features)

]
edges = [
    (src, dst, {"correlation": correlation_matrix[src, dst]}) 
    for src in range(n_features) 
    for dst in range(n_features)
    if \
        src < dst and
        np.abs(correlation_matrix[src, dst]) > 0] # only draw edges above a certain correlation threshold

def node_size(g, n, d):
#     r,p = d["correlation_y"]
#     return abs(r)*1000
#     return -np.log(p)
#     r = d['r_sq']
#     return abs(r)/10
    return d['p_value']
#     return (1000 if p==0 else -np.log(p))


def node_color(g, n, d):
    return d["label"]

def node_name(g, n, d):
    return d["name"]

def edge_color(g, src, dst, d):
    r = d["correlation"]*10
    return [0,0,0,abs(r)]

def edge_width(g, src, dst, d):
    """function to calculate edge width from edge properties"""
    return np.abs(d["correlation"])
#     return np.abs(d["correlation"])


fig1, ax = plt.subplots(1,1,figsize=(10,10))

# Selecting the axis-X making the bottom and top axes False.


nx_plot(
    nodes=nodes,
#     nodes_labels =node_name,
    nodes_pos=embeddings,
    edges=edges,
    nodes_args=dict(
        node_size=node_size,
        node_color=node_color
    )
    ,
    edges_args=dict(
        edge_color=edge_color
        , width=edge_width
    )
)






# In[ ]:


fig1.tight_layout()

#for the correlation network of nodes size being significance of the trends
fig1.savefig("../Figures/correlation_network_clusters_GA_effect.png", facecolor='white', dpi=300,  bbox_inches='tight')



# In[ ]:


label


# In[ ]:


nodes = [
    (i, {
#         "correlation_y": y_correlation[i],
#         "label": y_labels.iloc[i],
         "name": X.iloc[:,i].name,
#         "name":dict_rsq[i][0],
#     "r_sq": dict_rsq[i][1],
#        "r_sq": dict_rsq[i][1].p
#         "r_sq": p_values[i]
#         "p_value": df.iloc[:,i].log_10_p_value
        "p_value": df[X.iloc[:,i].name].adj_log_10,
        "label":colors_plt[label[i]]

    })
    for i in range(n_features)

]
edges = [
    (src, dst, {"correlation": correlation_matrix[src, dst]}) 
    for src in range(n_features) 
    for dst in range(n_features)
    if \
        src < dst and
        np.abs(correlation_matrix[src, dst]) > 0] # only draw edges above a certain correlation threshold

def node_size(g, n, d):
#     r,p = d["correlation_y"]
#     return abs(r)*1000
#     return -np.log(p)
#     r = d['r_sq']
#     return abs(r)/10
    return d['p_value']
#     return (1000 if p==0 else -np.log(p))


def node_color(g, n, d):

    return d["label"]


def node_name(g, n, d):
    if d["name"] in ['Neutrophils100leukocytesinBlood',\
                    'Basophils100leukocytesinBlood',\
                    'Lymphocytes100leukocytesinBlood',\
                    'Eosinophils100leukocytesinBlood',\
                    'Monocytes100leukocytesinBlood'] :
        return d["name"]

def edge_color(g, src, dst, d):
    r = d["correlation"]*10
    return [0,0,0,abs(r)]

def edge_width(g, src, dst, d):
    """function to calculate edge width from edge properties"""
    return np.abs(d["correlation"])
#     return np.abs(d["correlation"])


fig1, ax = plt.subplots(1,1,figsize=(15,15))


# Selecting the axis-X making the bottom and top axes False.


nx_plot(
    nodes=nodes,
    nodes_labels =node_name,
    nodes_pos=embeddings,
    edges=edges,
    nodes_args=dict(
        node_size=node_size,
        node_color=node_color
    )
    ,
    edges_args=dict(
        edge_color=edge_color
        , width=edge_width
    )
)
plt.show()



# In[ ]:


fig1.savefig("/Figures/correlation_network_clusters_CongAnom_effect.png", facecolor='white', dpi=300,  bbox_inches='tight')


# In[ ]:


nodes = [
    (i, {
#         "correlation_y": y_correlation[i],
#         "label": y_labels.iloc[i],
        "name": X.iloc[:,i].name,
#         "name":dict_rsq[i][0],
#     "r_sq": dict_rsq[i][1],
#        "r_sq": dict_rsq[i][1].p
#         "r_sq": p_values[i]
#         "p_value": df.iloc[:,i].log_10_p_value
        "p_value": df[X.iloc[:,i].name].adj_log_10,
        "label":colors_plt[label[i]]


    })
    for i in range(n_features)

]
edges = [
    (src, dst, {"correlation": correlation_matrix[src, dst]}) 
    for src in range(n_features) 
    for dst in range(n_features)
    if \
        src < dst and
        np.abs(correlation_matrix[src, dst]) > 0] # only draw edges above a certain correlation threshold

def node_size(g, n, d):
#     r,p = d["correlation_y"]
#     return abs(r)*1000
#     return -np.log(p)
#     r = d['r_sq']
#     return abs(r)/10
    return d['p_value']
#     return (1000 if p==0 else -np.log(p))


def node_color(g, n, d):
    return d["label"]

def node_name(g, n, d):
    if d['p_value']>=50:
        return d["name"]

def edge_color(g, src, dst, d):
    r = d["correlation"]*10
    return [0,0,0,abs(r)]

def edge_width(g, src, dst, d):
    """function to calculate edge width from edge properties"""
    return np.abs(d["correlation"])
#     return np.abs(d["correlation"])


fig1, ax = plt.subplots(1,1,figsize=(10,10))

# Selecting the axis-X making the bottom and top axes False.


nx_plot(
    nodes=nodes,
    nodes_labels =node_name,
    nodes_pos=embeddings,
    edges=edges,
    nodes_args=dict(
        node_size=node_size,
        node_color=node_color
    )
    ,
    edges_args=dict(
        edge_color=edge_color
        , width=edge_width
    )
)



# In[ ]:


fig.savefig("corr_network.pdf")


# In[ ]:


fig.savefig("disease_new_multivariate_icd9includ/corr_network.png")


# In[ ]:




