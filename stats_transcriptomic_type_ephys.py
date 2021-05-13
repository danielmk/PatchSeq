# -*- coding: utf-8 -*-
"""
Do statistics on the transcriptomic type and plot low-dimensional
embeddings.
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import ranksums

plt.rcParams["svg.fonttype"] = "none"
sns.set(
    context="paper",
    style="ticks",
    palette="colorblind",
    font="Arial",
    font_scale=2,
    color_codes=True,
)

# Load count and alignment data and merge them into one annotated dataframe
dirname = os.path.dirname(__file__)
ephys_path = os.path.join(dirname, "data", "ephys_full_df.csv")
ephys_df = pd.read_csv(ephys_path, index_col=0)
ephys_df = ephys_df[ephys_df.sequencing]
interneurons = ephys_df["PC vs IN Cluster"] == "IN"
coloc_bool = ephys_df["SST & Slc17a8 Positive"].astype(bool)
ephys_df["SST & Slc17a8 Positive"] = coloc_bool

fig = plt.figure()
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0:1])
ax2 = fig.add_subplot(gs[0, 1])
sns.scatterplot(
    x="SST Log2 CPM",
    y="Slc17a8 Log2 CPM",
    hue="SST & Slc17a8 Positive",
    data=ephys_df[interneurons],
    ax=ax1,
    s=150,
    linewidth=0,
    alpha=0.95,
)
ax1.set_aspect(
    (ax1.get_xlim()[1] - ax1.get_xlim()[0]) /
    (ax1.get_ylim()[1] - ax1.get_ylim()[0])
)
ax1.set_ylabel("Sst log2 CPM")
ax1.set_xlabel("Slc17a8 log2 CPM")

sns.scatterplot(
    x="TSNE 1",
    y="TSNE 2",
    hue="SST & Slc17a8 Positive",
    data=ephys_df[interneurons],
    alpha=0.95,
    s=150,
    ax=ax2,
    linewidth=0,
)
ax2.set_aspect(
    (ax2.get_xlim()[1] - ax2.get_xlim()[0]) /
    (ax2.get_ylim()[1] - ax2.get_ylim()[0])
)

"""TRANSGENIC AND TRANSCRIPTOMIC TYPE EPHYS"""
ephys_df["Coloc Genic Omic"] = False
coloc = ephys_df["SST & Slc17a8 Positive"]
ephys_df.loc[coloc, "Coloc Genic Omic"] = True
coloc_transgenic = (ephys_df["SST Positive"] &
                    (ephys_df["label"] == "VGlut3-EYFP"))
ephys_df.loc[coloc_transgenic, "Coloc Genic Omic"] = True

fig = plt.figure()
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0:1])
ax2 = fig.add_subplot(gs[0, 1])

sns.scatterplot(
    x="SST Log2 CPM",
    y="Slc17a8 Log2 CPM",
    hue="Coloc Genic Omic",
    data=ephys_df[interneurons],
    ax=ax1,
    s=150,
    linewidth=0,
    alpha=0.95,
)
ax1.set_aspect(
    (ax1.get_xlim()[1] - ax1.get_xlim()[0]) /
    (ax1.get_ylim()[1] - ax1.get_ylim()[0])
)
ax1.set_ylabel("Sst log2 CPM")
ax1.set_xlabel("Slc17a8 log2 CPM")

sns.scatterplot(
    x="TSNE 1",
    y="TSNE 2",
    hue="Coloc Genic Omic",
    data=ephys_df,
    alpha=0.95,
    s=150,
    ax=ax2,
    linewidth=0,
)
ax2.set_aspect(
    (ax2.get_xlim()[1] - ax2.get_xlim()[0]) /
    (ax2.get_ylim()[1] - ax2.get_ylim()[0])
)

# Analyze contingency between seq marker and fluorescent marker
seq_fluo_table = pd.crosstab(ephys_df["Transcriptomic Type"],
                             ephys_df["label"])

"""Ranksum test on electrophysiology for colocalizing cells"""
result_omic_dict = {}
features = [
    "Max. Freq. (Hz)",
    "Slow AHP (mV)",
    "Rheobase (pA)",
    "I at Max. Freq. (pA)",
    "Adaptation ratio",
    "Avg Spike Time (s)",
    "Input R (MOhm)",
    "Capacitance (pF)",
    "Sag Amplitude (mV)",
    "Resting (mV)",
    "RS AHP Amp. (mV)",
    "RS Max. Slope (mV/ms)",
    "RS Min. Slope (mV/ms)",
    "RS Peak (mV)",
    "RS Half Width (ms)",
    "RS Threshold (mV)",
    "LS AHP Amp. (mV)",
    "LS Max. Slope (mV/ms)",
    "LS Min. Slope (mV/ms)",
    "LS Peak (mV)",
    "LS Half Width (ms)",
    "LS Threshold (mV)",
]

coloc = ephys_df[ephys_df["SST & Slc17a8 Positive"]]
noncoloc = ephys_df[~ephys_df["SST & Slc17a8 Positive"]]
for f in features:
    x = coloc[f]
    y = noncoloc[f]
    result = ranksums(x, y)
    result_omic_dict[f] = [result.statistic, result.pvalue]

result_genicomic_dict = {}
coloc = ephys_df[ephys_df["Coloc Genic Omic"]]
noncoloc = ephys_df[~ephys_df["Coloc Genic Omic"]]
for f in features:
    x = coloc[f]
    y = noncoloc[f]
    result = ranksums(x, y)
    result_genicomic_dict[f] = [result.statistic, result.pvalue]
