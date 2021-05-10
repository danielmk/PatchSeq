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
full_df = pd.read_csv(r"C:\Users\Daniel\repos\PatchSeq\full_df.csv", index_col=0)
ephys_df = pd.read_csv(r"C:\Users\Daniel\repos\PatchSeq\ephys_df.csv", index_col=0)
adata.var_names_make_unique()
adata.obs_names_make_unique()

""" INTERNEURON MARKER CODE"""
# Marker Names
markers = ['Sst', 'Slc17a8', 'Vip', 'Pvalb', 'Cck', 'Npy', 'Calb2']

# adata_log = sc.pp.log1p(adata, base=2, copy=True)
adata.obs['SST Positive'] = (adata.obs_vector('Sst') > 0) & ~(adata.obs_vector('Cck') > 0)
adata.obs['Slc17a8 Positive'] = adata.obs_vector('Slc17a8') > 0
adata.obs['SST & Slc17a8 Positive'] = adata.obs['SST Positive'] & adata.obs['Slc17a8 Positive']
adata.obs['Cck Positive'] = ((adata.obs_vector('Cck') > 0) & ~(adata.obs_vector('Sst') > 0))

adata.obs['Transcriptomic Type'] = 'Other'
adata.obs['Transcriptomic Type'][adata.obs['SST Positive']] = "SST mRNA Positive"
adata.obs['Transcriptomic Type'][adata.obs['Cck Positive']] = "CCK mRNA Positive"

col_names = ['Max. Freq. (Hz)',
             'Rheobase (pA)',
             'I at Max. Freq. (pA)',
             'Adaptation ratio',
             'Avg Spike Time (s)']

# Merge full_df and adata.obs
df_merged = pd.concat([adata.obs, full_df], axis=1, sort=False)
adata.obs = pd.concat([adata.obs, full_df], axis=1, sort=False, join='inner')
ephys_merged = ephys_merged = pd.concat([adata.obs, ephys_df], axis=1, sort=False, join='inner')
df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
adata.obs = adata.obs.loc[:, ~adata.obs.columns.duplicated()]
ephys_merged = ephys_merged.loc[:, ~ephys_merged.columns.duplicated()]

ephys_merged_INs=ephys_merged[ephys_merged['PC vs IN Cluster'] == "IN"]

plot_ephys = True
if plot_ephys:
    for var_name in col_names:
        fig, ax = plt.subplots(1)
        sns.violinplot(x='Transcriptomic Type',
                       y=var_name,
                       data = df_merged,
                       kind='violin',
                       ax = ax,
                       saturation=0.1,
                       dodge=True,
                       order=["SST mRNA Positive",
                              "CCK mRNA Positive"])
        
        sns.swarmplot(x='Transcriptomic Type',
                      y=var_name,
                      data=df_merged,
                      dodge=True,
                      ax=ax,
                      alpha=0.8,
                      s=8,
                      order=["SST mRNA Positive",
                             "CCK mRNA Positive"])
