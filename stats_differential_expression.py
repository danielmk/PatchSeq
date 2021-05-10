# -*- coding: utf-8 -*-
"""
This script performs a differential expression analysis using the scanpy
package to determine which genes are differently expressed between two cell
types. The method uses the wilcoxon rank sum test and Benjamini-Hochberg
correction for multiple comparisons.
For another analysis a generalized linear model is fit to distinguish
the two cell types using a known set of marker genes. The result of fitting the
GLM can be used to identify which markers are useful for cell type
identification and after model validation it could be used in the future for
cell type prediction.
"""


import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

"""Data loading"""
# Load count and alignment data and merge them into one annotated dataframe
adata = sc.read_h5ad("count_exons_introns_full_named_postqc.h5ad")
ephys_df = pd.read_csv("ephys_df.csv", index_col=0)
full_df = pd.read_csv("full_df.csv", index_col=0)
annotations = pd.read_csv("annotations.csv", index_col=0, delimiter=';')

"""Data wrangling"""
adata.var_names_make_unique()
adata.obs.index = adata.obs['id']

# Delete ctrl samples
ctrl_sample_names = annotations.index[annotations['ctrl'] == 1]
ctrl_samples_adata = ctrl_sample_names.intersection(adata.obs.index)
non_ctrl_names = [name for name in adata.obs_names
                  if not name in ctrl_samples_adata]
adata = adata[non_ctrl_names, :]

# Delete non-INs
good_cells = (ephys_df['PC vs IN Cluster'] == 'IN')
good_cell_names = ephys_df.index[good_cells].intersection(adata.obs.index)
adata = adata[good_cell_names, :]

# Add the label feature to adata
adata.obs['label'] = full_df.loc[adata.obs.index, 'label']

"""Feature engineering and preprocessing"""
sst_pos = adata.obs_vector('Sst') > 0
slc17a8_pos = adata.obs_vector('Slc17a8') > 0
adata.obs['SST Positive'] = sst_pos
adata.obs['Slc17a8 Positive'] = slc17a8_pos
adata.obs['SST & Slc17a8 Positive'] = sst_pos & slc17a8_pos

t_type = 'Transcriptomic Type'
adata.obs[t_type] = 'Other'
adata.obs.loc[sst_pos, t_type] = "SST RNA Positive"
adata.obs.loc[slc17a8_pos, t_type] = "Slc17a8 RNA Positive"
adata.obs.loc[sst_pos & slc17a8_pos, t_type] = "SST & Slc17a8 RNA Positive"

coloc = 'SST & Slc17a8 Coloc'
adata.obs[coloc] = 'Other'
adata.obs[coloc][sst_pos & slc17a8_pos] = "SST & Slc17a8 Coloc"

# Take log2 of the count matrix
adata = sc.pp.log1p(adata, base=2, copy=True)
adata.obs["SST Exp Log2"] = adata.obs_vector('Sst')
adata.obs["Slc17a8 Exp Log2"] = adata.obs_vector('Slc17a8')

# Seaborn plot styling
plt.rcParams['svg.fonttype'] = 'none'
sns.set(context='paper',
        style='ticks',
        palette='colorblind',
        font='Arial',
        font_scale=2,
        color_codes=True)

# SST vs VGLUT3 Scatter
fig, ax = plt.subplots(1)
sns.scatterplot(x=adata.obs["SST Exp Log2"],
                y=adata.obs["Slc17a8 Exp Log2"],
                ax=ax,
                s=150,
                linewidth=0)
ax.set_xlabel("Sst Expr Log2")
ax.set_ylabel("Slc17a8 Expr Log2")

"""Differential expression analysis"""
sc.tl.rank_genes_groups(adata, 
                        groupby='SST & Slc17a8 Coloc',
                        method='wilcoxon',
                        key_added='diff_exp',
                        pts=True,
                        n_genes=50)

diff_genes = np.array(adata.uns['diff_exp']['names']['SST & Slc17a8 Coloc'],
                      dtype=np.str)
"""Differential expression analysis heatmap"""
sc.pl.heatmap(adata,
              var_names=diff_genes,
              groupby='SST & Slc17a8 Coloc')

"""Interneuron marker analysis"""
markers = ["Gad1", "Drd2", "Npy", 'Sst', "Chat", "Th", "Pvalb", "Htr3a",
           "Lhx6", "Tac1", "Cox6a2", "Sox11", "Slc17a8"]
# 'Igfbp4', 'Igfbpl1' kept out becauselow variance.
# Excluded Markers: ['Gpr88', 'D830015G02Rik', 'Adora2a', 'Drd1a', 'Pthlh',
# 'Chodl', 'Hhip', 'Mia', 'Slc5a7', 'Trh', 'Igfbp4', 'Igfbpl1']

fig, ax = plt.subplots(1)
stacked_violin = sc.pl.stacked_violin(adata,
                                      markers,
                                      groupby="Transcriptomic Type",
                                      stripplot=True,
                                      swap_axes=True,
                                      size=3,
                                      ax=ax)
plt.xticks(rotation=0)

"""Train GLM to distinguish colocalizing from other cells using the markers"""
adata_df = adata[:, markers].to_df()
features = sm.add_constant(adata_df)
classes = adata.obs["Transcriptomic Type"].copy()
classes[~(classes == "SST & Slc17a8 RNA Positive")] = "Other"
classes = classes.cat.remove_unused_categories()
classes = pd.Categorical(classes).rename_categories([0, 1])

binomial_model = sm.GLM(classes, features, family=sm.families.Binomial())
binomial_results = binomial_model.fit()

print(binomial_results.summary())

classes = classes.rename_categories(["Other", "SST & Slc17a8 RNA Positive"])
binomial_prediction = pd.DataFrame({"classes": classes,
                                    "prediction": binomial_results.predict()})

fig, ax = plt.subplots(1)
sns.swarmplot(x="classes",
              y ="prediction",
              data=binomial_prediction,
              ax=ax, s=10, alpha=0.8)
ax.set_ylabel("GLM Prediction")

"""Write the binomial GLM parameters to file"""
with open('glm_transcriptomic_type_output.csv','w') as f:
    f.write(binomial_results.summary().as_csv())
