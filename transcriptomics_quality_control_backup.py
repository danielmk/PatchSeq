# -*- coding: utf-8 -*-
"""
Perform quality control on samples and genes.

Quality control of samples includes total counts, number of genes, percentage
of mitochondrial genes and alignment rate.

Genes are excluded for 0 counts and low average counts.

Output of the script are several figures of the quality control parameters and
a file called count_exons_introns_full_named_postqc.h5ad. It is annotated,
does not contain the excluded cells/genes, is normalized to counts-per-million
(CPM) and log2 transformed.
"""

import os
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load count and alignment data and merge them into one annotated dataframe
dirname = os.path.dirname(__file__)
adata_path = os.path.join(dirname, 'data', 'count_exons_introns_full_named.h5ad')
adata = sc.read_h5ad(adata_path)

adata.obs.index = np.array([x.split('.')[0] for x in adata.obs.index], dtype = str)

alignment_path = os.path.join(dirname, 'data', 'mapping_stats.csv')
alignment = pd.read_csv(alignment_path,
                        delimiter=";", index_col=0)

alignment.index = np.array(alignment.index, dtype=np.str)
adata.obs = pd.concat([adata.obs, alignment], axis=1, sort=False)

adata.obs = adata.obs.set_index('id')

annotations_path = os.path.join(dirname, 'data', 'annotations.csv')
annotations = pd.read_csv(annotations_path,
                        delimiter=";", index_col=0)
adata.obs = pd.concat([adata.obs, annotations], axis=1, sort=False, join='inner')

"""Quality Control of Cells"""
sc.pp.calculate_qc_metrics(adata, inplace=True)

def mad_exclusion(v, crit=3):
    """Exclusion based on 3*median absolute deviation"""
    mad = (v - np.median(v)).abs().median()
    thr_mad = np.median(v) - crit * mad
    excluded = v < thr_mad

    return excluded, thr_mad

# Exclude based on small library size (2 * median absolute deviation)
total_counts_log10 = np.log10(adata.obs["total_counts"])
total_counts_bool, total_counts_thr = mad_exclusion(total_counts_log10)

# Exclude based on number of expressed genes
#n_genes_log10 = np.log10(adata.obs["n_genes_by_counts"])
n_genes_log10 = np.log10(adata.obs["n_genes_by_counts"])
n_genes_bool, n_genes_thr = mad_exclusion(n_genes_log10)

# Exclude if more than 10% of reads are mitochondrial (Carter et al. 2018)
gene_names = adata.var.index
mito_gene_names = gene_names[gene_names.str.startswith("mt-")]
adata.obs['total_mito_counts'] = adata.to_df()[mito_gene_names].sum(axis=1)
adata.obs['perc_mito_counts'] = (adata.obs['total_mito_counts'] / adata.obs["total_counts"]) * 100 

perc_mito_gene_counts = adata.obs['perc_mito_counts']
cutoff = 10.0
perc_mito_gene_counts_excl_bool = (perc_mito_gene_counts > cutoff)
perc_mito_gene_counts_excl_idc = adata.obs.index[perc_mito_gene_counts_excl_bool.values]

# Perc Aligned
exon_alignment_bool, exon_alignment_thr = mad_exclusion(adata.obs['perc_of_total_exon'])
intron_alignment_bool, intron_alignment_thr = mad_exclusion(adata.obs['perc_of_total_intron'])

# Exclude Negative Controls
bool_ctrl = adata.obs['ctrl'] == 1

# All excluded samples
all_excluded_samples_bool = (total_counts_bool | 
        n_genes_bool | 
        perc_mito_gene_counts_excl_bool | 
        exon_alignment_bool | 
        intron_alignment_bool |
        bool_ctrl)



"""Plotting for quality control"""
sns.set(context='paper',
        style='whitegrid',
        palette='colorblind',
        font='Arial',
        font_scale=2,
        color_codes=True)

fig, ax = plt.subplots(2,3)
sns.distplot(total_counts_log10, ax=ax[0,0], norm_hist=False, kde=False, bins=20)
ax[0,0].vlines(total_counts_thr, 0, 45, linestyle='dashed')
sns.distplot(n_genes_log10, ax=ax[0,1], norm_hist=False, kde=False, bins=10)
ax[0,1].vlines(n_genes_thr, 0, 45, linestyle='dashed')
sns.distplot(perc_mito_gene_counts, ax=ax[0,2], norm_hist=False, kde=False, bins=100)
ax[0,2].vlines(cutoff, 0, 50, linestyle='dashed')

ax[0,0].set_ylabel("# Cells")
ax[0,0].set_xlabel("Total Counts Log10")
ax[0,1].set_xlabel("Number of Genes Log10")
ax[0,2].set_xlabel("% Mitochondrial Counts")

sns.distplot(adata.obs['perc_of_total_exon'], ax=ax[1,0], norm_hist=False, kde=False, bins=40)
ax[1,0].vlines(exon_alignment_thr, 0, 20, linestyle='dashed')
sns.distplot(adata.obs['perc_of_total_intron'], ax=ax[1,1], norm_hist=False, kde=False, bins=40)
ax[1,1].vlines(intron_alignment_thr, 0, 20, linestyle='dashed')
ax[1,0].set_ylabel("# Cells")
ax[1,1].set_ylabel("# Cells")
ax[1,0].set_xlabel("% Exon Aligned")
ax[1,1].set_xlabel("% Intron Aligned")

"""Normalization"""
sc.pp.normalize_total(adata, target_sum=1e6, exclude_highly_expressed=True, inplace=True)

"""Quality Control of Genes"""
# Remove genes with zero total counts
zero_genes_bool = adata.var['total_counts'] == 0
zero_genes_idc = adata.var.index[zero_genes_bool.values]

# Remove genes that have a low average expression
mean_counts_log10 = np.log10(adata.var['mean_counts'][~zero_genes_bool].values)
low_mean_count_bool = mean_counts_log10 < 0
#adata.var['mean_counts_log10'] = np.log10(adata.var['mean_counts'].values)
low_mean_count_idc = adata.var[~zero_genes_bool].index[mean_counts_log10 < 0]

# Plotting gene 
sns.distplot(mean_counts_log10, ax=ax[1,2], norm_hist=False, kde=False, bins=40)
ax[1,2].vlines(0, 0, 2100, linestyle='dashed')
ax[1,2].set_ylabel("# Genes")
ax[1,2].set_xlabel("Mean Count Log10")

# Create the filtered AnnData object
adata = adata[~all_excluded_samples_bool]
adata_T = adata.copy().T[~zero_genes_bool]
adata_T = adata_T[~low_mean_count_bool]
adata = adata_T.copy().T

adata.var_names_make_unique()
adata.obs_names_make_unique()

adata = sc.pp.log1p(adata, base=2, copy=True)
adata_save_path = os.path.join(dirname, 'data',
                               'count_exons_introns_full_named_postqc.h5ad')
adata.write(adata_save_path)
