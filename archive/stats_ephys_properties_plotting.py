# -*- coding: utf-8 -*-
"""
This script walks through the electrophysiological raw data (.abf files) and
claculates 28 electrophysiological properties for each cell.
The passive properties:
    Input Resistance (MOhm)
    Capacitance (pF)
    Sag Amplitude (mV)
    Resting Potential (mV)

The active properties:
    Maximum frequency (Hz)
    Rheobase (pA)
    Slow After Hyperpolarization Amplitude (mV)
    Input at Maximum Frequency (pA)
    Adaptation Ratio
    Average Spike Time (s)

Single spike properties:
    Fast AHP Amplitude
    Maximum Slope
    Minimum Slope
    Peak
    Half-Width
    Threshold

The single spike properties are calculated separately for:
    First spike at rheobase
    First spike at maximum frequency
    Last spike at maximum frequency

After the electrophysiological properties are calculated, the top principal
components that explain 99% of the variance are calculated. Based on those pcs,
tSNE and UMAP embeddings are calculated and hierarchy linkage clustering is
performed.
This script writes its output to a .csv file and saves a variety of figures.

"""

import os
import numpy as np
import scipy.signal
import scipy.optimize
import scipy.spatial
import sklearn.manifold
import scipy.cluster.hierarchy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import statsmodels.api as sm

dirname = os.path.dirname(__file__)
ephys_path = os.path.join(dirname, 'data', 'ephys_features_df.csv')

ephys_df = pd.read_csv(
    ephys_path,
    delimiter=",",
    index_col = 0
)

col_names = [
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
    "FS AHP Amp. (mV)",
    "FS Max. Slope (mV/ms)",
    "FS Min. Slope (mV/ms)",
    "FS Peak (mV)",
    "FS Half Width (ms)",
    "FS Threshold (mV)",
    "LS AHP Amp. (mV)",
    "LS Max. Slope (mV/ms)",
    "LS Min. Slope (mV/ms)",
    "LS Peak (mV)",
    "LS Half Width (ms)",
    "LS Threshold (mV)"
]

"""PLOTTING"""
sns.set(
    context="paper",
    style="whitegrid",
    palette="colorblind",
    font="Arial",
    font_scale=2,
    color_codes=True,
)

"""SWARM WITH VIOLIN"""
for idx, var in enumerate(col_names[:6]):
    if idx % 3 == 0:
        fig, axs = plt.subplots(nrows=3)
    sns.violinplot(
        x="area_with_label",
        y=var,
        data=ephys_df,
        kind="violin",
        ax=axs[idx % 3],
        saturation=0.1,
        dodge=True,
        order=[
            "Unlabeled SO",
            "SST-EYFP SO",
            "VGlut3-EYFP SO",
            "VGlut3-EYFP SR",
            "Unlabeled SR",
            "Unlabeled PCL",
        ],
    )
    sns.swarmplot(
        x="area_with_label",
        y=var,
        data=ephys_df,
        dodge=True,
        ax=axs[idx % 3],
        alpha=0.8,
        order=[
            "Unlabeled SO",
            "SST-EYFP SO",
            "VGlut3-EYFP SO",
            "VGlut3-EYFP SR",
            "Unlabeled SR",
            "Unlabeled PCL",
        ],
        s=8,
    )
    # axs[idx%3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for x in axs:
        x.set_xlabel("")

fig, axs = plt.subplots(nrows=4)
for idx, var in enumerate(col_names[6:10]):
    sns.violinplot(
        x="area_with_label",
        y=var,
        data=ephys_df,
        kind="violin",
        ax=axs[idx],
        saturation=0.1,
        dodge=True,
        order=[
            "Unlabeled SO",
            "SST-EYFP SO",
            "VGlut3-EYFP SO",
            "VGlut3-EYFP SR",
            "Unlabeled SR",
            "Unlabeled PCL",
        ],
    )
    sns.swarmplot(
        x="area_with_label",
        y=var,
        data=ephys_df,
        dodge=True,
        ax=axs[idx],
        alpha=0.8,
        order=[
            "Unlabeled SO",
            "SST-EYFP SO",
            "VGlut3-EYFP SO",
            "VGlut3-EYFP SR",
            "Unlabeled SR",
            "Unlabeled PCL",
        ],
        s=8,
    )
    for x in axs:
        x.set_xlabel("")


for idx, var in enumerate(col_names[10:]):
    if idx % 3 == 0:
        fig, axs = plt.subplots(nrows=3)
    sns.violinplot(
        x="area_with_label",
        y=var,
        data=ephys_df,
        kind="violin",
        ax=axs[idx % 3],
        saturation=0.1,
        dodge=True,
        order=[
            "Unlabeled SO",
            "SST-EYFP SO",
            "VGlut3-EYFP SO",
            "VGlut3-EYFP SR",
            "Unlabeled SR",
            "Unlabeled PCL",
        ],
    )
    sns.swarmplot(
        x="area_with_label",
        y=var,
        data=ephys_df,
        dodge=True,
        ax=axs[idx % 3],
        alpha=0.8,
        order=[
            "Unlabeled SO",
            "SST-EYFP SO",
            "VGlut3-EYFP SO",
            "VGlut3-EYFP SR",
            "Unlabeled SR",
            "Unlabeled PCL",
        ],
        s=8,
    )
    for x in axs:
        x.set_xlabel("")

"""Use GLM to distinguish label types"""
unlabeled = ephys_df['label'] == 'Unlabeled'
features = ephys_df[col_names][~unlabeled]
features = sm.add_constant(features)
features = features.loc[:,col_names[0:16]]
features = sm.add_constant(features)
classes = pd.Categorical(ephys_df['label'][~unlabeled])
classes = classes.rename_categories([1,2])

poisson_model = sm.GLM(classes, features, family=sm.families.Poisson())
poisson_results = poisson_model.fit()

print(poisson_results.summary())

classes = classes.rename_categories(["SST-EYFP", "VGlut3-EYFP"])

poisson_out_df = pd.DataFrame({"classes": classes, "prediction": poisson_results.predict()})
sns.swarmplot(x="classes", y ="prediction", data=poisson_out_df, s = 15, alpha=0.9)



sns.set(
    context="paper",
    style="ticks",
    palette="colorblind",
    font="Arial",
    font_scale=2,
    color_codes=True,
)

order=["SST-EYFP",
       "VGlut3-EYFP"]

"""SWARM WITH VIOLIN"""
for idx, var in enumerate(col_names[:6]):
    if idx % 3 == 0:
        fig, axs = plt.subplots(nrows=3)
    sns.violinplot(
        x="label",
        y=var,
        data=ephys_df,
        kind="violin",
        ax=axs[idx % 3],
        saturation=0.1,
        dodge=True,
        order=order,
    )
    sns.swarmplot(
        x="label",
        y=var,
        data=ephys_df,
        dodge=True,
        ax=axs[idx % 3],
        alpha=0.8,
        order=order,
        s=8,
    )
    # axs[idx%3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for x in axs:
        x.set_xlabel("")

fig, axs = plt.subplots(nrows=4)
for idx, var in enumerate(col_names[6:10]):
    sns.violinplot(
        x="label",
        y=var,
        data=ephys_df,
        kind="violin",
        ax=axs[idx],
        saturation=0.1,
        dodge=True,
        order=order,
    )
    sns.swarmplot(
        x="label",
        y=var,
        data=ephys_df,
        dodge=True,
        ax=axs[idx],
        alpha=0.8,
        order=order,
        s=8,
    )
    for x in axs:
        x.set_xlabel("")


for idx, var in enumerate(col_names[10:]):
    if idx % 3 == 0:
        fig, axs = plt.subplots(nrows=3)
    sns.violinplot(
        x="label",
        y=var,
        data=ephys_df,
        kind="violin",
        ax=axs[idx % 3],
        saturation=0.1,
        dodge=True,
        order=order,
    )
    sns.swarmplot(
        x="label",
        y=var,
        data=ephys_df,
        dodge=True,
        ax=axs[idx % 3],
        alpha=0.8,
        order=order,
        s=8,
    )
    for x in axs:
        x.set_xlabel("")

