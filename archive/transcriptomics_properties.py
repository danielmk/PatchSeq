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

sns.set(context='paper',
        style='whitegrid',
        palette='colorblind',
        font='Arial',
        font_scale=2,
        color_codes=True)

# Load count and alignment data and merge them into one annotated dataframe
adata = sc.read_h5ad(r"E:\Dropbox\Dropbox\01_SST Project_daniel\033_PatchSeq CA3 SO\transcriptomics\Patch-Seq\count_exons_introns_full_named_postqc.h5ad")
ephys_df = pd.read_csv(r"E:\Dropbox\Dropbox\01_SST Project_daniel\033_PatchSeq CA3 SO\analysis_scripts\full_df.csv", index_col=0)
adata.obs = adata.obs.set_index('id')
adata.var_names_make_unique()
#ephys_df['seq_id'] = np.array(ephys_df['seq_id'], dtype=np.str)
#mapping_df = ephys_df['seq_id'].copy()
#mapping_df['seq_id'] = np.array(ephys_df['seq_id'], dtype=np.str)

""" INTERNEURON MARKER CODE"""
# Marker Names
markers = ['Sst', 'Slc17a8', 'Vip', 'Pvalb', 'Cck', 'Npy', 'Calb2']

# adata_log = sc.pp.log1p(adata, base=2, copy=True)
adata.obs['SST Positive'] = adata.obs_vector('Sst') > 0
adata.obs['Slc17a8 Positive'] = adata.obs_vector('Slc17a8') > 0
adata.obs['SST & Slc17a8 Positive'] = adata.obs['SST Positive'] & adata.obs['Slc17a8 Positive']

adata.obs['Transcriptomic Type'] = ''
adata.obs['Transcriptomic Type'][adata.obs['SST Positive']] = "SST RNA Positive"
adata.obs['Transcriptomic Type'][adata.obs['Slc17a8 Positive']] = "Slc17a8 RNA Positive"
adata.obs['Transcriptomic Type'][adata.obs['SST & Slc17a8 Positive']] = "SST & Slc17a8 RNA Positive"

adata.obs['SST & Slc17a8 Coloc'] = 'Other'
adata.obs['SST & Slc17a8 Coloc'][adata.obs['SST & Slc17a8 Positive']] = "SST & Slc17a8 Coloc"

# Merge ephys_df and adata.obs
df_merged = pd.concat([adata.obs, ephys_df], axis=1, sort=False)
adata.obs = pd.concat([adata.obs, ephys_df], axis=1, sort=False, join='inner')

col_names = ['Max. Freq. (Hz)',
             'Slow AHP (mV)',
             'Rheobase (pA)',
             'I at Max. Freq. (pA)',
             'Adaptation ratio',
             'Avg Spike Time (s)',
             'Input R (MOhm)',
             'Capacitance (pF)',
             'Sag Amplitude (mV)',
             'Resting (mV)',
             'RS AHP Amp. (mV)',
             'RS Max. Slope (mV/ms)',
             'RS Min. Slope (mV/ms)',
             'RS Peak (mV)',
             'RS Half Width (ms)',
             'RS Threshold (mV)',
             'FS AHP Amp. (mV)',
             'FS Max. Slope (mV/ms)',
             'FS Min. Slope (mV/ms)',
             'FS Peak (mV)',
             'FS Half Width (ms)',
             'FS Threshold (mV)',
             'LS AHP Amp. (mV)',
             'LS Max. Slope (mV/ms)',
             'LS Min. Slope (mV/ms)',
             'LS Peak (mV)',
             'LS Half Width (ms)',
             'LS Threshold (mV)',
             'Rheobase idx']

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
                       order=["SST RNA Positive",
                              "SST & Slc17a8 RNA Positive", 
                              "Slc17a8 RNA Positive"])
        
        sns.swarmplot(x='Transcriptomic Type',
                      y=var_name,
                      data=df_merged,
                      dodge=True,
                      ax=ax,
                      alpha=0.8,
                      s=8,
                      order=["SST RNA Positive",
                             "SST & Slc17a8 RNA Positive",
                             "Slc17a8 RNA Positive"])

# Genetic Expression
adata_log10 = sc.pp.log1p(adata, base=10, copy=True)
sc.tl.rank_genes_groups(adata_log10, 
                        groupby='SST & Slc17a8 Coloc',
                        method='wilcoxon',
                        key_added='diff_exp',
                        pts=True,
                        n_genes=50)

sc.pl.heatmap(adata_log10,
              var_names = np.array(adata_log10.uns['diff_exp']['names']['SST & Slc17a8 Coloc'], dtype=np.str),
              groupby='SST & Slc17a8 Coloc')

# Correlation between Transcriptomics and Transgenic type

fig, ax = plt.subplots(2,1)
sc.pl.violin(adata_log10,
             keys='Sst',
             groupby='area_with_label',
             multi_panel=True,
             size=5,
             order=['SST-EYFP SO',
                    'VGlut3-EYFP SO',
                    'Unlabeled SO',
                    'VGlut3-EYFP SR',
                    'Unlabeled SR',
                    'Unlabeled PCL'],
            ax=ax[0])

sc.pl.violin(adata_log10,
         keys='Slc17a8',
         groupby='area_with_label',
         multi_panel=True,
         size=5,
         order=['SST-EYFP SO',
                'VGlut3-EYFP SO',
                'Unlabeled SO',
                'VGlut3-EYFP SR',
                'Unlabeled SR',
                'Unlabeled PCL'],
        ax=ax[1])

for a in ax:
    a.set_ylim((0,5))

# Cluster the transcriptomic data
# sc.pp.highly_variable_genes(adata, n_top_genes=1001, flavor='cell_ranger')


"""
sst_exp, sst_exp_log = adata.obs_vector('Sst'), adata_log.obs_vector('Sst')
vglut3_exp, vglut3_exp_log = adata.obs_vector('Slc17a8'), adata_log.obs_vector('Slc17a8')
vip_exp, vip_exp_log = adata.obs_vector('Vip'), adata_log.obs_vector('Vip')
pvalb_exp, pvalb_exp_log = adata.obs_vector('Pvalb'), adata_log.obs_vector('Pvalb')
cck_exp, cck_exp_log = adata.obs_vector('Cck'), adata_log.obs_vector('Cck')
npy_exp, npy_exp_log = adata.obs_vector('Npy'), adata_log.obs_vector('Npy')
calb2_exp, calb2_exp_log = adata.obs_vector('Calb2'), adata_log.obs_vector('Calb2')

sst_nonzero = sst_exp > 0
vglut3_nonzero = vglut3_exp > 0
vip_nonzero = vip_exp > 0
pvalb_nonzero = pvalb_exp > 0
cck_nonzero = cck_exp > 0
npy_nonzero = npy_exp > 0
calb2_nonzero = calb2_exp > 0
"""
"""
fig, ax = plt.subplots(2, 1)
sns.distplot(sst_exp[sst_nonzero], bins=50, ax=ax[0])
sns.distplot(sst_exp_log[sst_nonzero], bins=50, ax=ax[1])
ax[0].set_title("Sst")
ax[0].set_ylabel("Normalized Density")
ax[0].set_xlabel("CPM")
ax[1].set_ylabel("Normalized Density")
ax[1].set_xlabel("CPM log2")

fig, ax = plt.subplots(2, 1)
sns.distplot(vglut3_exp[vglut3_nonzero], bins=50, ax=ax[0])
sns.distplot(vglut3_exp_log[vglut3_nonzero], bins=50, ax=ax[1])
ax[0].set_title("VGlut3")
ax[0].set_ylabel("Normalized Density")
ax[0].set_xlabel("CPM")
ax[1].set_ylabel("Normalized Density")
ax[1].set_xlabel("CPM log2")

fig, ax = plt.subplots(2, 1)
sns.distplot(vip_exp[vip_nonzero], bins=50, ax=ax[0])
sns.distplot(vip_exp_log[vip_nonzero], bins=50, ax=ax[1])
ax[0].set_title("VIP")
ax[0].set_ylabel("Normalized Density")
ax[0].set_xlabel("CPM")
ax[1].set_ylabel("Normalized Density")
ax[1].set_xlabel("CPM log2")

fig, ax = plt.subplots(2, 1)
sns.distplot(pvalb_exp[pvalb_nonzero], bins=50, ax=ax[0])
sns.distplot(pvalb_exp_log[pvalb_nonzero], bins=50, ax=ax[1])
ax[0].set_title("Pvalb")
ax[0].set_ylabel("Normalized Density")
ax[0].set_xlabel("CPM")
ax[1].set_ylabel("Normalized Density")
ax[1].set_xlabel("CPM log2")

fig, ax = plt.subplots(2, 1)
sns.distplot(cck_exp[cck_nonzero], bins=50, ax=ax[0])
sns.distplot(cck_exp_log[cck_nonzero], bins=50, ax=ax[1])
ax[0].set_title("CCK")
ax[0].set_ylabel("Normalized Density")
ax[0].set_xlabel("CPM")
ax[1].set_ylabel("Normalized Density")
ax[1].set_xlabel("CPM log2")

fig, ax = plt.subplots(2, 1)
sns.distplot(npy_exp[npy_nonzero], bins=50, ax=ax[0])
sns.distplot(npy_exp_log[npy_nonzero], bins=50, ax=ax[1])
ax[0].set_title("Npy")
ax[0].set_ylabel("Normalized Density")
ax[0].set_xlabel("CPM")
ax[1].set_ylabel("Normalized Density")
ax[1].set_xlabel("CPM log2")

fig, ax = plt.subplots(2, 1)
sns.distplot(calb2_exp[calb2_nonzero], bins=50, ax=ax[0])
sns.distplot(calb2_exp_log[calb2_nonzero], bins=50, ax=ax[1])
ax[0].set_title("Calb2")
ax[0].set_ylabel("Normalized Density")
ax[0].set_xlabel("CPM")
ax[1].set_ylabel("Normalized Density")
ax[1].set_xlabel("CPM log2")
"""