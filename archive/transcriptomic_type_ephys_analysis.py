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
from sklearn.preprocessing import StandardScaler
from matplotlib import gridspec

plt.rcParams['svg.fonttype'] = 'none'
sns.set(context='paper',
        style='ticks',
        palette='colorblind',
        font='Arial',
        font_scale=2,
        color_codes=True)

# Load count and alignment data and merge them into one annotated dataframe
adata = sc.read_h5ad(r"E:\Dropbox\Dropbox\01_SST Project_daniel\033_PatchSeq CA3 SO\transcriptomics\Patch-Seq\count_exons_introns_full_named_postqc.h5ad")
adata.obs.index = adata.obs['id']
full_df = pd.read_csv(r"C:\Users\Daniel\repos\PatchSeq\full_df.csv", index_col=0)
ephys_df = pd.read_csv(r"C:\Users\Daniel\repos\PatchSeq\ephys_df.csv", index_col=0)
adata.var_names_make_unique()
adata.obs_names_make_unique()

adata = sc.pp.log1p(adata, base=2, copy=True)

""" INTERNEURON MARKER CODE"""
# Marker Names
# markers = ['Sst', 'Slc17a8', 'Vip', 'Pvalb', 'Cck', 'Npy', 'Calb2']

# adata_log = sc.pp.log1p(adata, base=2, copy=True)
adata.obs['SST CPM'] = adata.obs_vector('Sst')
adata.obs['Slc17a8 CPM'] = adata.obs_vector('Slc17a8')
adata.obs['SST Positive'] = adata.obs_vector('Sst') > 0
adata.obs['Slc17a8 Positive'] = adata.obs_vector('Slc17a8') > 0
adata.obs['SST & Slc17a8 Positive'] = adata.obs['SST Positive'] & adata.obs['Slc17a8 Positive']
adata.obs['Sst CPM'] = adata.obs_vector('Sst')
adata.obs['Slc17a8 CPM'] = adata.obs_vector('Slc17a8')

adata.obs['Transcriptomic Type'] = 'Other'
adata.obs['Transcriptomic Type'][adata.obs['SST Positive']] = "SST RNA Positive"
adata.obs['Transcriptomic Type'][adata.obs['Slc17a8 Positive']] = "Slc17a8 RNA Positive"
adata.obs['Transcriptomic Type'][adata.obs['SST & Slc17a8 Positive']] = "SST & Slc17a8 RNA Positive"

adata.obs['SST & Slc17a8 Coloc'] = 'Other'
adata.obs['SST & Slc17a8 Coloc'][adata.obs['SST & Slc17a8 Positive']] = "SST & Slc17a8 Coloc"

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
             'LS Threshold (mV)']


# Merge full_df and adata.obs
df_merged = pd.concat([adata.obs, full_df], axis=1, sort=False)
adata.obs = pd.concat([adata.obs, full_df], axis=1, sort=False, join='inner')
ephys_merged = pd.concat([adata.obs, ephys_df],
                                        axis=1, sort=False, join='inner')
df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
adata.obs = adata.obs.loc[:, ~adata.obs.columns.duplicated()]
ephys_merged = ephys_merged.loc[:, ~ephys_merged.columns.duplicated()]

plot_ephys = False
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


# Correlation between Transcriptomics and Transgenic type
fig, ax = plt.subplots(2,1)
sc.pl.violin(adata,
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

sc.pl.violin(adata,
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
    a.set_ylim((0,15))
ax[0].set_ylabel("SST Log2 Expr")
ax[1].set_ylabel("Slc17a8 Log2 Expr")


fig = plt.figure()
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0,0:1])
ax2 = fig.add_subplot(gs[0,1])
ephys_merged_INs = ephys_merged[ephys_merged['PC vs IN Cluster'] == "IN"]

sns.scatterplot(x='Sst CPM',
                y='Slc17a8 CPM',
                hue='SST & Slc17a8 Coloc',
                data=ephys_merged_INs,
                ax=ax1,
                s=150,
                linewidth=0,
                alpha=0.95)
ax1.set_aspect((ax1.get_xlim()[1] - ax1.get_xlim()[0]) /
              (ax1.get_ylim()[1] - ax1.get_ylim()[0]))
ax1.set_ylabel('Sst log2 CPM')
ax1.set_xlabel('Slc17a8 log2 CPM')

sns.scatterplot(
    x="TSNE 1",
    y="TSNE 2",
    hue='SST & Slc17a8 Coloc',
    data=ephys_merged_INs,
    alpha=0.95,
    s=150,
    ax=ax2,
    hue_order=[
        "Other",
        "SST & Slc17a8 Coloc",
    ],
    linewidth=0,
)
ax2.set_aspect((ax2.get_xlim()[1] - ax2.get_xlim()[0]) /
              (ax2.get_ylim()[1] - ax2.get_ylim()[0]))


# GLM to classify colocalizing cells
adata_classifier = adata[adata.obs['ephys'] == 1]
adata_classifier.obs['SST & Slc17a8 Coloc'] = pd.CategoricalIndex(adata_classifier.obs['SST & Slc17a8 Coloc'])

features_unscaled = sm.add_constant(adata_classifier.obs[col_names])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_unscaled)
features = pd.DataFrame(features_scaled,
                        index=features_unscaled.index,
                        columns=features_unscaled.columns)

classes = adata_classifier.obs['SST & Slc17a8 Coloc'].cat.rename_categories([1,2])

poisson_model = sm.GLM(classes, features, family=sm.families.Poisson())
poisson_results = poisson_model.fit()

print(poisson_results.summary())

poisson_out_df = pd.DataFrame({"classes": classes, "prediction": poisson_results.predict()})
fig, ax = plt.subplots(1)
sns.swarmplot(x="classes", y ="prediction", data=poisson_out_df)

f = open('glm_transcriptomic_type_ephys_output.csv','w')
f.write(poisson_results.summary().as_csv())
f.close()


col_names.append("Transcriptomic Type")
descriptive_stats = adata.obs[col_names].groupby("Transcriptomic Type")
descriptive_stats = descriptive_stats.describe()
descriptive_stats.T.to_csv("ephys_transcriptomic_type_stats.csv")

"""SST FALSE POSITIVE ANALYSIS"""
pcl_sst = (adata.obs['area'] == 'PCL') & (adata.obs['SST Positive'] == True)

"""TRANSGENIC AND TRANSCRIPTOMIC TYPE EPHYS"""
ephys_merged_INs['Coloc Genic Omic'] = False
coloc = ephys_merged_INs['SST & Slc17a8 Coloc'] == 'SST & Slc17a8 Coloc'
ephys_merged_INs.loc[coloc, 'Coloc Genic Omic'] = True
coloc_transgenic = ephys_merged_INs['SST Positive'] & (ephys_merged_INs['label'] == 'VGlut3-EYFP')
ephys_merged_INs.loc[coloc_transgenic, 'Coloc Genic Omic'] = True

fig = plt.figure()
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0,0:1])
ax2 = fig.add_subplot(gs[0,1])

sns.scatterplot(x='Sst CPM',
                y='Slc17a8 CPM',
                hue='Coloc Genic Omic',
                data=ephys_merged_INs,
                ax=ax1,
                s=150,
                linewidth=0,
                alpha=0.95)
ax1.set_aspect((ax1.get_xlim()[1] - ax1.get_xlim()[0]) /
              (ax1.get_ylim()[1] - ax1.get_ylim()[0]))
ax1.set_ylabel('Sst log2 CPM')
ax1.set_xlabel('Slc17a8 log2 CPM')

sns.scatterplot(
    x="TSNE 1",
    y="TSNE 2",
    hue='Coloc Genic Omic',
    data=ephys_merged_INs,
    alpha=0.95,
    s=150,
    ax=ax2,
    linewidth=0,
)
ax2.set_aspect((ax2.get_xlim()[1] - ax2.get_xlim()[0]) /
              (ax2.get_ylim()[1] - ax2.get_ylim()[0]))

# Analyze contingency between seq marker and fluorescent marker

seq_fluo_table = pd.crosstab(ephys_merged_INs['Transcriptomic Type'],
                             ephys_merged_INs['label'])

#adata.obs.to_csv("trans_df.csv")

"""HCN CHANNEL ANALYSIS"""
"""
adata_log2.obs['Hcn1 CPM'] = adata_log2.obs_vector('Hcn1')
adata_log2.obs['Hcn2 CPM'] = adata_log2.obs_vector('Hcn2')
adata_log2.obs['Hcn3 CPM'] = adata_log2.obs_vector('Hcn3')
adata_log2.obs['Hcn4 CPM'] = adata_log2.obs_vector('Hcn4')
adata_log2.obs['Hcn Total CPM'] = adata_log2.obs['Hcn1 CPM']+adata_log2.obs['Hcn2 CPM']+adata_log2.obs['Hcn3 CPM']+adata_log2.obs['Hcn4 CPM']
fig, ax = plt.subplots(1)
sns.violinplot(
    x="label",
    y='Hcn Total CPM',
    data=adata_log2.obs,
    kind="violin",
    ax=ax,
    saturation=0.1,
    dodge=True,
    order=[
        "SST-EYFP",
        "VGlut3-EYFP"],
)
sns.swarmplot(
    x="label",
    y='Hcn Total CPM',
    data=adata_log2.obs,
    dodge=True,
    ax=ax,
    alpha=0.8,
    order=[
        "SST-EYFP",
        "VGlut3-EYFP"],
    s=8,
)

sns.scatterplot(x=adata_log2.obs['Sag Amplitude (mV)'], y=adata_log2.obs['Hcn4 CPM'])
"""

