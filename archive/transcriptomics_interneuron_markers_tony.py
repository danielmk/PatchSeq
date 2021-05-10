# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 20:46:53 2020

@author: Daniel
"""


import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

# Load count and alignment data and merge them into one annotated dataframe
adata = sc.read_h5ad(r"E:\Dropbox\Dropbox\01_SST Project_daniel\033_PatchSeq CA3 SO\transcriptomics\Patch-Seq\count_exons_introns_full_named_postqc.h5ad")
full_df = pd.read_csv(r"C:\Users\Daniel\repos\PatchSeq\full_df.csv", index_col=0)
ephys_df = pd.read_csv(r"C:\Users\Daniel\repos\PatchSeq\ephys_df.csv", index_col=0)
adata.var_names_make_unique()
adata.obs_names_make_unique()

adata.obs = pd.concat([adata.obs, full_df], axis=1, sort=False, join='inner')
adata.obs = adata.obs.loc[:, ~adata.obs.columns.duplicated()]

adata = adata[adata.obs_names[adata.obs['ephys'] == 1],:]
ephys_df['seq_id'] = np.array(ephys_df['seq_id'], dtype=np.str)
adata.obs['PC vs IN Cluster'] = ephys_df.loc[adata.obs.index]['PC vs IN Cluster']
#mapping_df = ephys_df['seq_id'].copy()
#mapping_df['seq_id'] = np.array(ephys_df['seq_id'], dtype=np.str)

""" INTERNEURON MARKER CODE"""
# Marker Names
adata.obs['SST Positive'] = adata.obs_vector('Sst') > 0
adata.obs['Slc17a8 Positive'] = adata.obs_vector('Slc17a8') > 0
adata.obs['SST & Slc17a8 Positive'] = adata.obs['SST Positive'] & adata.obs['Slc17a8 Positive']

adata.obs['Transcriptomic Type'] = 'Other'
adata.obs['Transcriptomic Type'][adata.obs['SST Positive']] = "SST RNA Positive"
adata.obs['Transcriptomic Type'][adata.obs['Slc17a8 Positive']] = "Slc17a8 RNA Positive"
adata.obs['Transcriptomic Type'][adata.obs['SST & Slc17a8 Positive']] = "SST & Slc17a8 RNA Positive"

adata.obs['SST & Slc17a8 Coloc'] = 'Other'
adata.obs['SST & Slc17a8 Coloc'][adata.obs['SST & Slc17a8 Positive']] = "SST & Slc17a8 Coloc"

adata_log = sc.pp.log1p(adata, base=2, copy=True)
adata_log.obs["SST Exp Log2"] = adata_log.obs_vector('Sst')
adata_log.obs["Slc17a8 Exp Log2"] = adata_log.obs_vector('Slc17a8')

# Seaborn Styling
plt.rcParams['svg.fonttype'] = 'none'
sns.set(context='paper',
        style='whitegrid',
        palette='colorblind',
        font='Arial',
        font_scale=2,
        color_codes=True)

# SST vs VGLUT3 Scatter
x = adata_log.obs_vector('Sst')
y = adata_log.obs_vector('Slc17a8')
adata.obs['Kcnt1 Expr'] = adata_log.obs_vector('Kcnt1')

fig, ax = plt.subplots(1)
sns.scatterplot(x=x, y=y, ax=ax, s=150, linewidth=0)
ax.set_xlabel("Sst Expr Log2")
ax.set_ylabel("Slc17a8 Expr Log2")

# Genetic Expression
#adata_log10 = sc.pp.log1p(adata, base=10, copy=True)
sc.tl.rank_genes_groups(adata_log, 
                        groupby='SST & Slc17a8 Coloc',
                        method='wilcoxon',
                        key_added='diff_exp',
                        pts=True,
                        n_genes=50)

sc.pl.heatmap(adata_log,
              var_names = np.array(adata_log.uns['diff_exp']['names']['SST & Slc17a8 Coloc'], dtype=np.str),
              groupby='SST & Slc17a8 Coloc')

# Check for interneuron marker expression differences
markers = ["Gad1", "Drd2",
           "Npy", 'Sst', "Chat", "Pvalb", "Htr3a", "Lhx6", 
           "Tac1", "Cox6a2", "Sox11", "Slc17a8", 'Kcnt1']
# Excluded Markers: ['Gpr88', 'D830015G02Rik', 'Adora2a', 'Drd1a', 'Pthlh', 
# 'Chodl', 'Hhip', 'Mia', 'Slc5a7', 'Trh', 'lgfbp4', 'lgfbpl1']

stacked_violin = sc.pl.stacked_violin(adata_log, markers, groupby="Transcriptomic Type", stripplot=True, swap_axes=True, size=3)
plt.xticks(rotation=0)

# Run GLM
adata_df = adata_log.to_df()
features = sm.add_constant(adata_df[markers])
classes = adata_log.obs["Transcriptomic Type"].copy()
classes[~(classes == "SST & Slc17a8 RNA Positive")] = "Other"
classes = classes.cat.remove_unused_categories()
classes = pd.Categorical(classes).rename_categories([0,1])

poisson_model = sm.GLM(classes, features, family=sm.families.Binomial())
poisson_results = poisson_model.fit()

print(poisson_results.summary())

classes= classes.rename_categories(["Other", "SST & Slc17a8 RNA Positive"])
poisson_out_df = pd.DataFrame({"classes": classes, "prediction": poisson_results.predict()})

fig, ax = plt.subplots(1)
sns.swarmplot(x="classes", y ="prediction", data=poisson_out_df, ax=ax, s=10, alpha=0.8)
ax.set_ylabel("GLM Prediction")
f = open('glm_transcriptomic_type_output.csv','w')
f.write(poisson_results.summary().as_csv())
f.close()


"""Interleukin Stuff for Tony"""
"""
markers = ["Gad1", "Npy", 'Sst', "Chat", "Th", "Pvalb", "Slc17a8", "Il18", "Il18r1"]
# Excluded Markers: ['Gpr88', 'D830015G02Rik', 'Adora2a', 'Drd1a', 'Pthlh', 
# 'Chodl', 'Hhip', 'Mia', 'Slc5a7', 'Trh', 'lgfbp4', 'lgfbpl1']

stacked_violin = sc.pl.stacked_violin(adata_log, markers, groupby="Transcriptomic Type", stripplot=True, swap_axes=True, size=3)
plt.xticks(rotation=0)

# Run GLM
adata_df = adata_log.to_df()
features = sm.add_constant(adata_df[markers])
classes = adata_log.obs["Transcriptomic Type"].copy()
classes[~(classes == "SST & Slc17a8 RNA Positive")] = "Other"
classes = classes.cat.remove_unused_categories()
classes = pd.Categorical(classes).rename_categories([0,1])

poisson_model = sm.GLM(classes, features, family=sm.families.Binomial())
poisson_results = poisson_model.fit()

print(poisson_results.summary())

classes= classes.rename_categories(["Other", "SST & Slc17a8 RNA Positive"])
poisson_out_df = pd.DataFrame({"classes": classes, "prediction": poisson_results.predict()})

fig, ax = plt.subplots(1)
sns.swarmplot(x="classes", y ="prediction", data=poisson_out_df, ax=ax, s=10, alpha=0.8)
ax.set_ylabel("GLM Prediction")
"""
"""
# GLM on all classes
adata_df = adata_log.to_df()
features = sm.add_constant(adata_df[markers])
classes = adata_log.obs["Transcriptomic Type"].copy()
classes = pd.Categorical(classes).rename_categories([1,2,3,4])

poisson_model = sm.GLM(classes, features, family=sm.families.Poisson())
poisson_results = poisson_model.fit()

print(poisson_results.summary())

poisson_out_df = pd.DataFrame({"classes": classes, "prediction": poisson_results.predict()})

sns.swarmplot(x="classes", y ="prediction", data=poisson_out_df)
"""