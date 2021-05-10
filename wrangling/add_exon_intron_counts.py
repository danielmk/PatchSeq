# -*- coding: utf-8 -*-
"""
Add exon and intron counts and write a .csv of the result.
"""

import pandas as pd
import numpy as np

exon_data = (r"E:\Dropbox\Dropbox\01_SST Project_daniel\033_PatchSeq CA3 SO"
             r"\transcriptomics\Patch-Seq\count_exons_counts_full.csv")

intron_data = (r"E:\Dropbox\Dropbox\01_SST Project_daniel\033_PatchSeq CA3 SO"
               r"\transcriptomics\Patch-Seq\count_introns_counts_full.csv")

exon_df = pd.read_csv(exon_data, sep=';', index_col=0)

intron_df = pd.read_csv(intron_data, sep=';', index_col=0)

exon_df.index = [x.split('.')[0] for x in exon_df.index]

intron_df.index = [x.split('.')[0] for x in intron_df.index]

exon_intron_df = exon_df.add(intron_df, fill_value=0).astype(np.int)

exon_intron_df.to_csv(r"E:\Dropbox\Dropbox\01_SST Project_daniel\033_PatchSeq "
                      r"CA3 SO\transcriptomics\Patch-Seq"
                      "\exons_introns_counts_full.csv")
