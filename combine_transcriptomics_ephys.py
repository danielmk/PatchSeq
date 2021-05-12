# -*- coding: utf-8 -*-
"""
This script adds trancriptomic information to the ephys dataframe and
ephys information to the transcriptomic annotated data.

The ephys data is in ephys_features_annotated_df.csv and the transcriptomics
data is in count_exons_introns_full_named_postqc.h5ad

The outputs are two files: ephys_full_df.csv and trans_anndf.h5ad
One file is not sufficient because some samples are missing transcriptomics
or ephys and the anndata structure does not handle missing values well.
"""

import os
import pandas as pd
import scanpy as sc
import numpy as np

# Load count and alignment data and merge them into one annotated dataframe
dirname = os.path.dirname(__file__)
trans_path = os.path.join(dirname, "data",
                          "count_exons_introns_full_named_postqc.h5ad")
trans_adata = sc.read_h5ad(trans_path)
trans_adata.obs.index = np.array(
    [x.split(".")[0] for x in trans_adata.obs.index], dtype=str
)

ephys_path = os.path.join(dirname, "data", "ephys_features_annotated_df.csv")
ephys_df = pd.read_csv(ephys_path, index_col=0)

"""Enrich the annotation of trans_data"""
trans_adata.obs["SST Log2 CPM"] = trans_adata.obs_vector("Sst")
trans_adata.obs["Slc17a8 Log2 CPM"] = trans_adata.obs_vector("Slc17a8")
sst_pos = trans_adata.obs["SST Log2 CPM"] > 0
trans_adata.obs["SST Positive"] = sst_pos
slc17a8_pos = trans_adata.obs["Slc17a8 Log2 CPM"] > 0
trans_adata.obs["Slc17a8 Positive"] = slc17a8_pos

trans_adata.obs["SST & Slc17a8 Positive"] = sst_pos & slc17a8_pos

trans_adata.obs["Transcriptomic Type"] = "Other"
trans_adata.obs["Transcriptomic Type"][sst_pos] = "SST RNA Positive"
trans_adata.obs["Transcriptomic Type"][slc17a8_pos] = "Slc17a8 RNA Positive"
coloc = sst_pos & slc17a8_pos
trans_adata.obs["Transcriptomic Type"][coloc] = "SST & Slc17a8 RNA Positive"

"""Comine ephys and transgenic"""
unique_cols = ephys_df.columns.difference(trans_adata.obs.columns)
row_intersect = ephys_df.index.intersection(trans_adata.obs.index)
trans_adata = trans_adata[row_intersect]
trans_adata.obs = pd.concat(
    [trans_adata.obs, ephys_df[unique_cols]], axis=1, sort=False, join="inner"
)

unique_cols = trans_adata.obs.columns.difference(ephys_df.columns)
ephys_df = pd.concat(
    [ephys_df, trans_adata.obs[unique_cols]], axis=1, sort=False, join="outer"
)

ephys_df = ephys_df[ephys_df.ephys]

"""Save the combined data structures"""
adata_save_path = os.path.join(dirname, "data", "trans_anndf.h5ad")
trans_adata.write(adata_save_path)

ephys_save_path = os.path.join(dirname, "data", "ephys_full_df.csv")
ephys_df.to_csv(ephys_save_path)
