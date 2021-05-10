# -*- coding: utf-8 -*-
"""
Convert ensembl names to gene names.
"""


import pandas as pd
import scanpy as sc

data_path = (r"E:\Dropbox\Dropbox\01_SST Project_daniel\033_PatchSeq CA3 SO"
             r"\transcriptomics\Patch-Seq\count_exons_introns_counts_full.csv")

tdata = sc.read_text(data_path, delimiter=",")
tdata = tdata.transpose()

name_path = (r"E:\Dropbox\Dropbox\01_SST Project_daniel\033_PatchSeq CA3 SO"
             r"\transcriptomics\Patch-Seq\transcript_names_sorted.txt")

row_names = open(name_path).read().split('\n')[0:-1]

tdata.var.index = row_names

tdata.write_h5ad(r"E:\Dropbox\Dropbox\01_SST Project_daniel\033_PatchSeq CA3 SO"
                 r"\transcriptomics\Patch-Seq\count_exons_introns_full_named.h5ad")
