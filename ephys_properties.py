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
performed. This clustering identifies two clusters. In one of them, all samples
that were collected in the pyramidal cell layer are located. The cells in this
putative pyramidal cell cluster are excluded from further analysis. The same
clustering as described above is repeated only for the putative interneurons.

This script writes its output to a .csv file that contains the
electrophysiological properties and the embeddings for the interneurons.
Furthermore, several figures are created to illustrate the embeddings and
principal components.

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

"""Load annotations"""
annotations_path = (
    r"E:\Dropbox\Dropbox\01_SST Project_daniel" r"\033_PatchSeq CA3 SO\annotations.csv"
)
dtypes = {
    "sample_id": np.str,
    "seq_id": np.str,
    "ephys": np.bool,
    "cDNA": np.bool,
    "sequencing": np.bool,
    "batch": np.int,
    "label": np.str,
    "area": np.str,
    "resting_potential": np.float,
}
annotations = pd.read_csv(
    annotations_path,
    delimiter=";",
    dtype=dtypes,
)

annotations = annotations.rename(index=annotations["sample_id"])
annotations["area_with_label"] = annotations["label"] + " " + annotations["area"]

"""FIND ALL ABF FILES"""
parent = r"E:\Dropbox\Dropbox\01_SST Project_daniel\033_PatchSeq CA3 SO"
# Descend down the directory tree and find all .abf files
all_abf_files = []
for root, dirs, files in os.walk(parent):
    for name in files:
        if ".abf" in name and not "excluded" in name:
            all_abf_files.append(root + "\\" + name)

all_abf_files = [x for x in all_abf_files if "active" in x or "passive" in x]
active_abf_files = [x for x in all_abf_files if "active" in x]
passive_abf_files = [x for x in all_abf_files if "passive" in x]

"""DEFINE SOME PARAMETERS FOR FOLLOWING CALCULATIONS"""
step_start = 2.0781 * pq.s  # Time when the current injection starts
step_stop = 3.0781 * pq.s  # Time when current injection stops
delay = 0.05 * pq.s  # The first AP occuring within 50 ms of current step onset
extr_left = -0.002 * pq.s  # Time to extract left of AP thr
extr_right = 0.0025 * pq.s  # Time to extract right of AP thr
current_delta = 20  # Difference between current inj. in pA
volt_threshold = 10 * pq.mV  # Voltage threshold for AP detection
dvdt_threshold = 10  # AP threshold detection threshold in mV/ms
ljp = 16  # liquid junction potential in mV

stats_dict = {}  # The statistics dictionary will contain results

"""HELPERS"""


def exp_decay(t, tau, V):
    return V * np.e ** (-t / tau)


def linear_function(x, a, b):
    return a * x + b


"""Create the dataframe"""
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
    "LS Threshold (mV)",
    "Rheobase idx",
]

result_df = pd.DataFrame(columns=col_names, dtype=np.float)

"""Create the dictionary containing the important traces"""
trace_dict = {}

"""Loop through all sample_id and populate result_df"""
for x in annotations["sample_id"]:
    print("Key: " + x)
    # If a sample key does not show up as a file key, it's all nans
    # if not x in file_keys:
    if not annotations.loc[x]["ephys"]:
        continue

    trace_dict[x] = {}
    path_active = list(filter(lambda k: (x in k), active_abf_files))[0]
    # block = data_dict_active[x][0]  # The current data block
    block_active = AxonIO(path_active).read_block(signal_group_mode="split-all")
    # Calculate the indices of current step onset and offset
    sr = block_active.segments[0].analogsignals[0].sampling_rate
    si = (1 / sr).magnitude * 1000
    stim_start_idx = int((step_start * sr).magnitude)
    stim_stop_idx = int((step_stop * sr).magnitude)

    # Find AP time stamps at the 0mV threshold crossings in each sweep
    threshold_crossings_idc = []  # Indices
    threshold_crossings_n = []  # Number
    threshold_crossings_ts = []  # Time stamps
    for sweep in block_active.segments:
        threshold_idc = np.where(sweep.analogsignals[0] > volt_threshold)[0]
        threshold_crossing_idc = threshold_idc[
            np.argwhere(np.diff(threshold_idc) > 1)[:, 0] + 1
        ]
        if threshold_idc.size > 0:
            threshold_crossing_idc = np.insert(
                threshold_crossing_idc, 0, threshold_idc[0]
            )
        thr = threshold_crossing_idc
        thr = np.delete(
            thr, np.argwhere(thr < stim_start_idx)
        )  # Exclude spontaneous spikes outside the current steps
        thr = np.delete(thr, np.argwhere(thr > stim_stop_idx))
        threshold_crossings_n.append(thr.size)
        threshold_crossings_idc.append(thr)
        threshold_crossings_ts.append(thr / sr)

    # Calculate maximum firing frequency
    n_spikes = np.array(threshold_crossings_n)
    max_freq_idx = n_spikes.argmax()
    step_start_idx = int((step_start * sr).magnitude)
    step_stop_idx = int((step_stop * sr).magnitude)
    avg_len = int((0.1 * pq.s * sr).magnitude)
    max_freq = n_spikes[max_freq_idx]
    max_freq_volt_trace = block_active.segments[max_freq_idx].analogsignals[0]

    # Calculate the rheobase
    current_amps = np.arange(0, 600, current_delta)
    rheobase_idx = np.where(n_spikes > 0)[0][0]
    rheobase = current_amps[rheobase_idx]

    # Extract the waveform of the first spike within delay (50 ms)
    for sweep_idx, sweep in enumerate(threshold_crossings_ts):
        if np.any(sweep < (delay + step_start)):
            spike_idx = int((sweep[0] * sr).magnitude)
            l_idx = int((extr_left * sr).magnitude)
            r_idx = int((extr_right * sr).magnitude)
            trace = block_active.segments[sweep_idx].analogsignals[0]
            curr_spike = trace[spike_idx + l_idx : spike_idx + r_idx]
            curr_spike_grad = np.gradient(np.array(curr_spike)[:, 0], si)
            break

    # Extract the waveform of the first spike at rheobase
    rheobase_ts = threshold_crossings_ts[rheobase_idx]
    curr_idx = int((rheobase_ts[0] * sr).magnitude)
    left_idx = int((extr_left * sr).magnitude)
    right_idx = int((extr_right * sr).magnitude)
    trace = block_active.segments[rheobase_idx].analogsignals[0]
    rheo_spike = trace[curr_idx + l_idx : curr_idx + r_idx]

    # Extract the waveform of the first spike at max frequency
    max_ts = threshold_crossings_ts[max_freq_idx]
    curr_idx = int((max_ts[0] * sr).magnitude)
    left_idx = int((extr_left * sr).magnitude)
    right_idx = int((extr_right * sr).magnitude)
    trace = max_freq_volt_trace
    first_max_spike = trace[curr_idx + l_idx : curr_idx + r_idx]

    # Extract the waveform of the last spike at max frequency
    max_ts = threshold_crossings_ts[max_freq_idx]
    curr_idx = int((max_ts[-1] * sr).magnitude)
    left_idx = int((extr_left * sr).magnitude)
    right_idx = int((extr_right * sr).magnitude)
    trace = max_freq_volt_trace
    last_max_spike = trace[curr_idx + left_idx : curr_idx + right_idx]

    spikes = [curr_spike, rheo_spike, first_max_spike, last_max_spike]
    spikes_label = ["cs", "rs", "fs", "ls"]

    spikes_dict = {}
    for idx, s in enumerate(spikes):
        trace_dict[x][spikes_label[idx]] = np.array(s)[:, 0]
        spikes_dict[spikes_label[idx]] = {}
        # Calculate the voltage threshold
        first_order_gradient = np.gradient(
            np.array(s)[:, 0], np.array(s.sampling_period.magnitude) * 1000
        )
        smooth_first_order_gradient = np.convolve(
            np.ones(10, "d"), first_order_gradient, mode="same"
        )
        thr_idx = np.where(smooth_first_order_gradient > dvdt_threshold)[0][0]
        spike_threshold = s[thr_idx]
        spikes_dict[spikes_label[idx]]["spike_threshold"] = spike_threshold
        # Fast AHP amplitude (from threshold)
        minimum = s.min()
        fast_ahp_amp = minimum - spike_threshold
        spikes_dict[spikes_label[idx]]["fast_ahp_amp"] = fast_ahp_amp
        # Max spike slope in mV/ms
        spike_max_slope = smooth_first_order_gradient.max()
        spikes_dict[spikes_label[idx]]["spike_max_slope"] = spike_max_slope
        # Min Spike slope in mV/ms
        spike_min_slope = smooth_first_order_gradient.min()
        spikes_dict[spikes_label[idx]]["spike_min_slope"] = spike_min_slope
        # AP peak (overshoot)
        spike_peak = s.max()
        spikes_dict[spikes_label[idx]]["spike_peak"] = spike_peak
        # AP half-width (between threshold and peak)
        # TODO MAKE HALFWIDTH CALCULATION MORE ROBUST TO FAST SPIKING
        half_height = ((spike_peak - spike_threshold) / 2.0) + spike_threshold
        left_of_peak = s[: np.argmax(curr_spike)]
        right_of_peak = s[np.argmax(curr_spike) :]
        left_idx = np.abs(left_of_peak - half_height).argmin()
        right_idx = np.abs(right_of_peak - half_height).argmin()
        time_left = left_of_peak.times[left_idx]
        time_right = right_of_peak.times[right_idx]
        half_width = time_right.rescale(pq.ms) - time_left.rescale(pq.ms)
        spikes_dict[spikes_label[idx]]["half_width"] = half_width

    # Calculate slow after hyperpolarization
    baseline = max_freq_volt_trace[step_start_idx - avg_len : step_start_idx].mean()
    after_step = max_freq_volt_trace[step_stop_idx : step_stop_idx + avg_len].mean()
    slow_ahp_amp = after_step - baseline
    # Calculate max freq current
    max_freq_current = current_amps[max_freq_idx]
    # Adaptation Ratio
    isis = np.diff(threshold_crossings_ts[max_freq_idx])
    adaptation_ratio = (isis[-1] / isis[0]).magnitude
    # Average spike time
    avg_spike_time = threshold_crossings_ts[max_freq_idx].mean()
    avg_spike_time = (avg_spike_time - step_start).magnitude

    """PASSIVE PROPERTIES ANALYSIS"""
    fits = []
    neg_peaks = []
    steady_states = []
    baselines = []
    # fit_inits = (0.025, -70)
    path_passive = list(filter(lambda k: (x in k), passive_abf_files))[0]
    block_passive = AxonIO(path_passive).read_block(signal_group_mode="split-all")
    for sweep in block_passive.segments[1:]:
        bl = sweep.analogsignals[0][stim_start_idx - avg_len : stim_start_idx].mean()
        baselines.append(bl)
        neg_peak = sweep.analogsignals[0][stim_start_idx:stim_stop_idx].min()
        neg_peaks.append(neg_peak)
        neg_peak_idx = sweep.analogsignals[0][stim_start_idx:stim_stop_idx].argmin()
        if neg_peak_idx > 5000:
            neg_peak_idx = 5000
        steady_state = sweep.analogsignals[0][
            stim_stop_idx - avg_len : stim_stop_idx
        ].mean()
        steady_states.append(steady_state)
        decay = sweep.analogsignals[0][
            stim_start_idx + 50 : stim_start_idx + neg_peak_idx
        ]
        decay = decay - decay.min()
        xdata = np.arange(len(decay)) * (1 / sr.magnitude)
        ydata = np.array(decay)[:, 0]
        fit = scipy.optimize.curve_fit(exp_decay, xdata, ydata, p0=(0.015, ydata[0]))
        fits.append(fit[0])
        ydatafit = exp_decay(xdata, fit[0][0], fit[0][1])

    decay_tau = np.median(np.array([x[0] for x in fits[1:]])) * 1000
    volt_changes = np.array(neg_peaks) - np.array(baselines)
    current_steps = np.arange(0, -len(volt_changes), -1) * current_delta
    neg_peaks = np.array(neg_peaks)
    sag_amplitudes = np.array(neg_peaks) - np.array(steady_states)
    if np.any(neg_peaks < -100):
        sag_idx = np.argwhere(neg_peaks < -100)[0][0]
    else:
        sag_idx = len(sag_amplitudes) - 1
    sag_amplitude = sag_amplitudes[sag_idx]
    sag_trace = block_passive.segments[sag_idx].analogsignals[0]

    # Ohms law: R = V/I
    line_fit = scipy.optimize.curve_fit(linear_function, current_steps, volt_changes)
    input_resistance = line_fit[0][0] * 1000  # *1000 converts to MegaOhm
    capacitance = (decay_tau / input_resistance) * 1000  # in microfarads

    # Find the resting membrane potential from a list
    resting_potential = annotations.loc[x]["resting potential"]

    r = np.array(
        [
            max_freq,
            slow_ahp_amp.magnitude,
            rheobase,
            max_freq_current,
            adaptation_ratio,
            avg_spike_time,
            input_resistance,
            capacitance,
            sag_amplitude,
            resting_potential,
            spikes_dict["rs"]["fast_ahp_amp"].magnitude,
            spikes_dict["rs"]["spike_max_slope"],
            spikes_dict["rs"]["spike_min_slope"],
            spikes_dict["rs"]["spike_peak"].magnitude - ljp,
            spikes_dict["rs"]["half_width"].magnitude,
            spikes_dict["rs"]["spike_threshold"].magnitude - ljp,
            spikes_dict["fs"]["fast_ahp_amp"].magnitude,
            spikes_dict["fs"]["spike_max_slope"],
            spikes_dict["fs"]["spike_min_slope"],
            spikes_dict["fs"]["spike_peak"].magnitude - ljp,
            spikes_dict["fs"]["half_width"].magnitude,
            spikes_dict["fs"]["spike_threshold"].magnitude - ljp,
            spikes_dict["ls"]["fast_ahp_amp"].magnitude,
            spikes_dict["ls"]["spike_max_slope"],
            spikes_dict["ls"]["spike_min_slope"],
            spikes_dict["ls"]["spike_peak"].magnitude - ljp,
            spikes_dict["ls"]["half_width"].magnitude,
            spikes_dict["ls"]["spike_threshold"].magnitude - ljp,
            rheobase_idx,
        ]
    )

    curr_df = pd.DataFrame(r[np.newaxis, :], columns=col_names, index=[x])
    result_df = result_df.append(curr_df)

    trace_dict[x]["sag_trace"] = np.array(sag_trace)[:, 0]
    trace_dict[x]["rheo_trace"] = np.array(
        block_active.segments[rheobase_idx].analogsignals[0]
    )[:, 0]
    trace_dict[x]["max_freq_trace"] = np.array(
        block_active.segments[max_freq_idx].analogsignals[0]
    )[:, 0]
    trace_dict[x]["time"] = np.arange(len(sag_trace)) * si
    trace_dict[x]["spikes_dict"] = spikes_dict

result_df = result_df.astype(np.float)
full_df = pd.concat([result_df, annotations], axis=1)
ephys_df = full_df[full_df['ephys']]


ephys_df_clustering = ephys_df[col_names[:-1]]
row_names_clustering = ephys_df_clustering.index
scaler = sklearn.preprocessing.MinMaxScaler()
ephys_df_clustering_scaled = scaler.fit_transform(ephys_df_clustering)
ephys_df_clustering_scaled = pd.DataFrame(
    ephys_df_clustering_scaled, columns=col_names[:-1], index=row_names_clustering
)

"""SEABORNE STYLING"""
sns.set(
    context="paper",
    style="whitegrid",
    palette="colorblind",
    font="Arial",
    font_scale=2,
    color_codes=True,
)


"""Find top principal components"""
pca = sklearn.decomposition.PCA(n_components=ephys_df_clustering_scaled.shape[1],
                                random_state=3456234)
pca_output = pca.fit_transform(ephys_df_clustering_scaled)
explained_variance = pca.explained_variance_ratio_.cumsum()*100

n_top_components = np.where(explained_variance > 99.0)[0][0]

fig, ax = plt.subplots(1)
ax.plot(np.arange(1, ephys_df_clustering_scaled.shape[1]+1, 1),
        explained_variance,
        marker='o')
ax.hlines(99, 1,ephys_df_clustering_scaled.shape[1]+1, linestyles='dashed')
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
tsne_df_annotated = pd.concat([tsne_df, annotations], axis=1)

fig, ax = plt.subplots(1)
sns.scatterplot(
    x="TSNE 1",
    y="TSNE 2",
    hue="area",
    data=tsne_df_annotated,
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
ax.plot(np.arange(1, ephys_df_clustering_scaled.shape[1]+1, 1),
        explained_variance,
        marker='o')
ax.hlines(99, 1,ephys_df_clustering_scaled.shape[1]+1, linestyles='dashed')
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
ephys_df = pd.concat([ephys_df, tsne_df], axis=1)
ephys_df = pd.concat([ephys_df, umap_df], axis=1)
ephys_df['area_with_label'] = ephys_df['label'] + ' ' + ephys_df['area']

"""Save the final ephys_df"""
ephys_df.to_csv("ephys_df.csv")

"""Do some plotting of the raw traces"""
"""Raw traces"""
keys = ["D70"]
fig, ax = plt.subplots()

start = 2000
for x in keys:
    ax.plot(trace_dict[x]["time"]-start, np.array(trace_dict[x]["sag_trace"]), alpha=0.8)
    ax.plot(trace_dict[x]["time"]-start, trace_dict[x]["max_freq_trace"], alpha=0.8)
    ax.plot(trace_dict[x]["time"]-start, trace_dict[x]["rheo_trace"], alpha=0.8)
    ax.set_xlim((0, 3500-start))
    ax.legend(("Minimum Voltage", "Maximum Frequency", "Rheobase"))
    ax.set_title("SST-EYFP")
    plt.xlabel("time (ms)")
    plt.ylabel("Voltage (mV)")

keys = ["M61"]
fig, ax = plt.subplots()

for x in keys:
    ax.plot(trace_dict[x]["time"]-start, np.array(trace_dict[x]["sag_trace"]), alpha=0.8)
    ax.plot(trace_dict[x]["time"]-start, trace_dict[x]["max_freq_trace"], alpha=0.8)
    ax.plot(trace_dict[x]["time"]-start, trace_dict[x]["rheo_trace"], alpha=0.8)
    ax.set_xlim((0, 3500-start))
    ax.legend(("Minimum Voltage", "Maximum Frequency", "Rheobase"))
    ax.set_title("VGlut3-EYFP")
    plt.xlabel("time (ms)")
    plt.ylabel("Voltage (mV)")

keys = ["D49"]
fig, ax = plt.subplots()

for x in keys:
    ax.plot(trace_dict[x]["time"]-start, np.array(trace_dict[x]["sag_trace"]), alpha=0.8)
    ax.plot(trace_dict[x]["time"]-start, trace_dict[x]["max_freq_trace"], alpha=0.8)
    ax.plot(trace_dict[x]["time"]-start, trace_dict[x]["rheo_trace"], alpha=0.8)
    ax.set_xlim((0, 3500-start))
    ax.legend(("Minimum Voltage", "Maximum Frequency", "Rheobase"))
    ax.set_title("Putative PC")
    plt.xlabel("time (ms)")
    plt.ylabel("Voltage (mV)")

