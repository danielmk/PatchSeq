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

plt.rcParams['svg.fonttype'] = 'none'
sns.set(context='paper',
        style='whitegrid',
        palette='colorblind',
        font='Arial',
        font_scale=2,
        color_codes=True)

# Load count and alignment data and merge them into one annotated dataframe
adata = sc.read_h5ad(r"E:\Dropbox\Dropbox\01_SST Project_daniel\033_PatchSeq CA3 SO\transcriptomics\Patch-Seq\count_exons_introns_full_named_postqc.h5ad")
ephys_df = pd.read_csv(r"C:\Users\Daniel\repos\PatchSeq\full_df.csv", index_col=0)
adata.var_names_make_unique()
#ephys_df['seq_id'] = np.array(ephys_df['seq_id'], dtype=np.str)
#mapping_df = ephys_df['seq_id'].copy()
#mapping_df['seq_id'] = np.array(ephys_df['seq_id'], dtype=np.str)

adata.obs['SST Positive'] = adata.obs_vector('Sst') > 0
adata.obs['Slc17a8 Positive'] = adata.obs_vector('Slc17a8') > 0
adata.obs['SST & Slc17a8 Positive'] = adata.obs['SST Positive'] & adata.obs['Slc17a8 Positive']

adata.obs['Transcriptomic Type'] = 'Other'
adata.obs['Transcriptomic Type'][adata.obs['SST Positive']] = "SST RNA Positive"
adata.obs['Transcriptomic Type'][adata.obs['Slc17a8 Positive']] = "Slc17a8 RNA Positive"
adata.obs['Transcriptomic Type'][adata.obs['SST & Slc17a8 Positive']] = "SST & Slc17a8 RNA Positive"

adata.obs['SST & Slc17a8 Coloc'] = 'Other'
adata.obs['SST & Slc17a8 Coloc'][adata.obs['SST & Slc17a8 Positive']] = "SST & Slc17a8 Coloc"

# Merge ephys_df and adata.obs
df_merged = pd.concat([adata.obs, ephys_df], axis=1, sort=False, join='inner')
adata.obs = pd.concat([adata.obs, ephys_df], axis=1, sort=False, join='inner')
df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
adata.obs = adata.obs.loc[:, ~adata.obs.columns.duplicated()]

"""Clustering Transcriptomic Data"""
# Exclude putative pyramidal cells
adata = adata[~(adata.obs['area_with_label'] == "Unlabeled PCL")]

# Calculate 1000 best genes
sc.pp.highly_variable_genes(adata, n_top_genes=3001, flavor='cell_ranger')

highly_variable_var_names = adata.var_names[adata.var.highly_variable]

# adata_log10 = sc.pp.log1p(adata, base=10, copy=True)

# Calculate t-SNE
n_pcas = 22  # 21 PCs explain >99% of variance
sc.tl.pca(adata, n_comps=n_pcas, random_state=390780)
sc.tl.tsne(adata, n_pcs=n_pcas, perplexity=5, early_exaggeration=12,
           learning_rate=10, random_state=3434645)



# Plot t-SNE
fig, ax = plt.subplots()
sc.pl.tsne(adata, color="area_with_label", ax=ax)
ax.set_aspect((ax.get_xlim()[1] - ax.get_xlim()[0]) /
              (ax.get_ylim()[1] - ax.get_ylim()[0]))

fig, ax = plt.subplots()
sc.pl.tsne(adata, color="Transcriptomic Type", ax=ax)
ax.set_aspect((ax.get_xlim()[1] - ax.get_xlim()[0]) /
              (ax.get_ylim()[1] - ax.get_ylim()[0]))
#sns.scatterplot(adata.obsm['X_tsne'][:,0], adata.obsm['X_tsne'][:,1])

# Calculate UMAP
sc.pp.neighbors(adata, n_pcs=n_pcas, random_state=2760789, metric="euclidean")
sc.tl.umap(adata)

fig, ax = plt.subplots()
sc.pl.umap(adata, color="area_with_label", ax=ax)
ax.set_aspect((ax.get_xlim()[1] - ax.get_xlim()[0]) /
              (ax.get_ylim()[1] - ax.get_ylim()[0]))

fig, ax = plt.subplots()
sc.pl.umap(adata, color="Transcriptomic Type", ax=ax)
ax.set_aspect((ax.get_xlim()[1] - ax.get_xlim()[0]) /
              (ax.get_ylim()[1] - ax.get_ylim()[0]))
