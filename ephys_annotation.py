# -*- coding: utf-8 -*-
"""
Annotates ephys_features_df with annotations.csv and calculates
TSNE and UMAP low-dimensional embeddings. Also generates several
figures for the TSNE embedding with and without pyramidal cells.
"""

import os
import numpy as np
from neo.io import AxonIO
import quantities as pq
import scipy.signal
import scipy.optimize
import scipy.spatial
import sklearn.manifold
import sklearn.preprocessing
import sklearn.decomposition
import scipy.cluster.hierarchy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import scipy.stats
import matplotlib as mpl
from matplotlib import gridspec

"""Load annotations"""
dirname = os.path.dirname(__file__)
annotations_path = os.path.join(dirname, 'data', 'annotations.csv')
ephys_path = os.path.join(dirname, 'data', 'ephys_features_df.csv')

dtypes = {
    "sample_id": str,
    "seq_id": str,
    "ephys": bool,
    "cDNA": bool,
    "sequencing": bool,
    "batch": int,
    "label": str,
    "area": str,
    "resting_potential": float,
}

annotations = pd.read_csv(
    annotations_path,
    delimiter=";",
    dtype=dtypes,
)

annotations = annotations.rename(index=annotations["sample_id"])
annotations["area_with_label"] = annotations["label"] + " " + annotations["area"]

ephys_df = pd.read_csv(
            ephys_path,
            delimiter=",",
            dtype=dtypes,
            index_col=0
        )

col_names = ephys_df.columns
annotated_df = pd.concat([ephys_df, annotations], axis=1, join='inner')

"""Preprocessing for dimensionality reduction"""
ephys_dr_df = ephys_df[col_names[:-1]].dropna()
row_names_clustering = ephys_dr_df.index
scaler = sklearn.preprocessing.MinMaxScaler()
ephys_dr_df_scaled = scaler.fit_transform(ephys_dr_df)
ephys_dr_df__scaled = pd.DataFrame(
    ephys_dr_df_scaled, columns=col_names[:-1], index=row_names_clustering
)

"""SEABORNE STYLING"""
plt.rcParams['svg.fonttype'] = 'none'
mpl.rcParams.update({'font.size': 12})
sns.set(
    context="paper",
    style="ticks",
    palette="colorblind",
    font="Arial",
    font_scale=1,
    color_codes=True,
)

"""Find top principal components"""
pca = sklearn.decomposition.PCA(n_components=ephys_dr_df_scaled.shape[1],
                                random_state=3456234)
pca_output = pca.fit_transform(ephys_dr_df_scaled)
explained_variance = pca.explained_variance_ratio_.cumsum()*100

n_top_components = np.where(explained_variance > 99.0)[0][0]

fig, ax = plt.subplots(1)
ax.plot(np.arange(1, ephys_dr_df_scaled.shape[1]+1, 1),
        explained_variance,
        marker='o',
        markersize=8)
ax.hlines(99, 1,ephys_dr_df_scaled.shape[1]+1, linestyles='dashed')
ax.set_xlabel("principal component")
ax.set_ylabel("% explained variance")
ax.set_title("Principal Components All Cells")

top_components = pca_output[:,:n_top_components]

"""Perform tSNE on the top components"""
tsne = sklearn.manifold.TSNE(
    n_components=2,
    perplexity=10,
    learning_rate=20,
    n_iter=10000,
    random_state=10,
    verbose=1,
    method="exact",
)
tsne_output = tsne.fit_transform(top_components)
tsne_df = pd.DataFrame(
    tsne_output, columns=["TSNE 1", "TSNE 2"], index=row_names_clustering
)
tsne_df_annotated_all_cells = pd.concat([tsne_df, annotations], axis=1)

fig, ax = plt.subplots(1)
sns.scatterplot(
    x="TSNE 1",
    y="TSNE 2",
    hue="area",
    data=tsne_df_annotated_all_cells,
    alpha=0.95,
    s=150,
    ax=ax,
    hue_order=[
        "SO",
        "SR",
        "PCL",
    ],
    linewidth=0
)
ax.set_aspect((ax.get_xlim()[1] - ax.get_xlim()[0]) /
              (ax.get_ylim()[1] - ax.get_ylim()[0]))

"""Perform UMAP on the top components"""
umap_inst = umap.UMAP(n_neighbors=15, random_state=45857823)
umap_output = umap_inst.fit_transform(top_components)
umap_df = pd.DataFrame(
    umap_output, columns=["UMAP 1", "UMAP 2"], index=row_names_clustering
)
umap_df_annotated = pd.concat([umap_df, annotations], axis=1)
fig, ax = plt.subplots()
sns.scatterplot(
    x="UMAP 1",
    y="UMAP 2",
    hue="area",
    data=umap_df_annotated,
    alpha=0.95,
    s=150,
    ax=ax,
    hue_order=[
        "SO",
        "SR",
        "PCL",
    ],
    linewidth=0
)
ax.set_aspect((ax.get_xlim()[1] - ax.get_xlim()[0]) /
              (ax.get_ylim()[1] - ax.get_ylim()[0]))

"""Perform hierarchy linkage clustering on tSNE"""
ephys_df_clustering_pdist = scipy.spatial.distance.pdist(
    tsne_output, metric="euclidean"
)
ephys_df_clustering_linkage = scipy.cluster.hierarchy.linkage(
    ephys_df_clustering_pdist, "single"
)
fig, ax = plt.subplots()
dn = scipy.cluster.hierarchy.dendrogram(ephys_df_clustering_linkage)
ax.set_ylabel("Distance")

cluster_labels = scipy.cluster.hierarchy.fcluster(ephys_df_clustering_linkage,
                                                 t=2,
                                                 criterion="maxclust")



cluster_df = pd.DataFrame(cluster_labels,
                          columns=["PC vs IN Clusters"], 
                          index=row_names_clustering)

cluster_categorical = pd.Categorical(cluster_labels)
cluster_categorical.rename_categories(["PC", "IN"], inplace=True)
cluster_df = pd.DataFrame(cluster_categorical,
                          index=ephys_df.index,
                          columns=["PC vs IN Cluster"])

ephys_df = pd.concat([ephys_df, cluster_df], axis=1)
annotated_df = pd.concat([annotated_df, cluster_df], axis=1)

"""Repeat embedding and clustering for INs only"""
in_df = ephys_df[ephys_df["PC vs IN Cluster"] == "IN"]

in_df_clustering = in_df[col_names[:-1]]
row_names_clustering = in_df_clustering.index
scaler = sklearn.preprocessing.MinMaxScaler()
in_df_clustering_scaled = scaler.fit_transform(in_df_clustering)
in_df_clustering_scaled = pd.DataFrame(
    in_df_clustering_scaled, columns=col_names[:-1], index=row_names_clustering
)


pca = sklearn.decomposition.PCA(n_components=in_df_clustering_scaled.shape[1],
                                random_state=3456234)
pca_output = pca.fit_transform(in_df_clustering_scaled)
explained_variance = pca.explained_variance_ratio_.cumsum()*100

n_top_components = np.where(explained_variance > 99.0)[0][0]

fig, ax = plt.subplots(1)
ax.plot(np.arange(1, ephys_dr_df_scaled.shape[1]+1, 1),
        explained_variance,
        marker='o',
        markersize=8)
ax.hlines(99, 1,ephys_dr_df_scaled.shape[1]+1, linestyles='dashed')
ax.set_xlabel("principal component")
ax.set_ylabel("% explained variance")
ax.set_title("Principal Components INs only")

top_components = pca_output[:,:n_top_components]

"""Perform tSNE on the top components"""
tsne = sklearn.manifold.TSNE(
    n_components=2,
    perplexity=10,
    learning_rate=20,
    n_iter=10000,
    random_state=10,
    verbose=1,
    method="exact",
)

tsne_output = tsne.fit_transform(top_components)
tsne_df = pd.DataFrame(
    tsne_output, columns=["TSNE 1", "TSNE 2"], index=row_names_clustering
)
tsne_df_annotated = pd.concat([tsne_df, annotations], axis=1)

fig, ax2 = plt.subplots()
sns.scatterplot(
    x="TSNE 1",
    y="TSNE 2",
    hue="label",
    data=tsne_df_annotated,
    alpha=0.95,
    s=150,
    ax=ax2,
    hue_order=[
        "Unlabeled",
        "SST-EYFP",
        "VGlut3-EYFP",
    ],
    linewidth=0
)
ax2.set_aspect((ax2.get_xlim()[1] - ax2.get_xlim()[0]) /
              (ax2.get_ylim()[1] - ax2.get_ylim()[0]))

"""Perform UMAP on the top components"""
umap_inst = umap.UMAP(n_neighbors=15, random_state=45857823)
umap_output = umap_inst.fit_transform(top_components)
umap_df = pd.DataFrame(
    umap_output, columns=["UMAP 1", "UMAP 2"], index=row_names_clustering
)

umap_df_annotated = pd.concat([umap_df, annotations], axis=1, join="inner")
fig, ax2 = plt.subplots()
sns.scatterplot(
    x="UMAP 1",
    y="UMAP 2",
    hue="label",
    data=umap_df_annotated,
    alpha=0.95,
    s=150,
    ax=ax2,
    hue_order=[
        "Unlabeled",
        "SST-EYFP",
        "VGlut3-EYFP",
    ],
    linewidth=0
)

ax2.set_aspect((ax2.get_xlim()[1] - ax2.get_xlim()[0]) /
              (ax2.get_ylim()[1] - ax2.get_ylim()[0]))

"""Perform hierarchy linkage clustering on tSNE"""
in_df_clustering_pdist = scipy.spatial.distance.pdist(
    tsne_output, metric="euclidean"
)
in_df_clustering_linkage = scipy.cluster.hierarchy.linkage(
    in_df_clustering_pdist, "single"
)
fig, ax = plt.subplots(1)
dn = scipy.cluster.hierarchy.dendrogram(in_df_clustering_linkage)
ax.set_ylabel("Distance")

"""Merge the ephys_df with the embeddings"""
annotated_df = pd.concat([annotated_df, tsne_df], axis=1)
#ephys_df = pd.concat([ephys_df, umap_df], axis=1)
annotated_df['area_with_label'] = annotated_df['label'] + ' ' + annotated_df['area']

"""Save the annotated df"""
save_path = os.path.join(dirname, 'data', 'ephys_features_annotated_df.csv')
annotated_df.to_csv(save_path)

