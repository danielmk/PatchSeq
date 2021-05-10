# PatchSeq
An analysis pipeline for patch seq data, a combination of patch-clamp electrophysiology and single cell sequencing. The data is not yet publicly available.

# Pipeline desription
1. ephys_feature_extraction.py

   Calculates electrophysiological features from raw recordings. Since this is the only script that loads raw recordings, it also generates example figures. The features are saved as ephys_features_df.csv

2. ephys_annotation.py

   Annotates the electrophysiological features with metadata and coordinates of a t-SNE embedding. This script also clusters samples into putative pyramidal cells and interneurons  based on their electrophysiological properties. Output is ephys_features_annotated_df.csv

3. transcriptomics_quality_control.py

   Quality control on samples and genes using scanpy package. Also normalizes read counts to Log2 counts-per-million and annotates metadata. Output is    count_exons_introns_full_named_postqc.h5ad

4. combine_transcriptomics_ephys.py

   Annotate electrophysiological with transcriptomic data and vice-versa. The output must be two data structures because not all recorded samples were sequenced and missing rows must be avoided in scanpy's annotated dataframe. Outputs are ephys_full_df.csv and trans_anndf.h5ad

5. stats_ephys_transgenic_type.py

   Statistically tests whether transgenically defined interneuron types are significantly different regarding their electrophysiological features. Does not implement multiple comparison correction yet.
 
6. stats_transcriptomic_type_ephys.py

   Statistically tests whether transcriptomically defined interneuron types are significantly differen regarding their electrophysiological features. Transcriptomic cell types are defined by their CPM. Does not implement multiple comparison correction yet.

7. stats_differential_expression.py
   Differential expression analysis to identify marker genes of a specific transcriptomic cell type.
