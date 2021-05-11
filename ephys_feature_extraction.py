# -*- coding: utf-8 -*-
'''
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

This script writes its output to a ephys_features_df.csv file.
'''

import os
import numpy as np
from neo.io import AxonIO
import quantities as pq
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

'''Load annotations'''
dirname = os.path.dirname(__file__)
annotations_path = os.path.join(dirname, "data", "annotations.csv")

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
annotations["area_with_label"] = (annotations["label"] + " " +
                                  annotations["area"])

'''Find all raw .abf files'''
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

'''Define parameters for calculation'''
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


def exp_decay(t, tau, V):
    '''Exponential decay helper function'''
    return V * np.e ** (-t / tau)


def line(x, a, b):
    '''Line helper function'''
    return a * x + b


result_df = pd.DataFrame()  # Initialize empty df for result storage

trace_dict = {}  # Store traces for later plotting

'''Loop through all sample_id and populate result_df'''
for x in annotations["sample_id"]:
    print("Key: " + x)
    feature_dict = {}
    # If a sample key does not show up as a file key, it's all nans
    # if not x in file_keys:
    if not annotations.loc[x]["ephys"]:
        continue

    trace_dict[x] = {}
    path_active = list(filter(lambda k: (x in k), active_abf_files))[0]
    # block = data_dict_active[x][0]  # The current data block
    abff = AxonIO(path_active)
    block_active = abff.read_block(signal_group_mode="split-all")
    # Calculate the indices of current step onset and offset
    sr = block_active.segments[0].analogsignals[0].sampling_rate
    si = (1 / sr).magnitude * 1000
    stim_start = int((step_start * sr).magnitude)
    stim_stop = int((step_stop * sr).magnitude)

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
            thr, np.argwhere(thr < stim_start)
        )  # Exclude spontaneous spikes outside the current steps
        thr = np.delete(thr, np.argwhere(thr > stim_stop))
        threshold_crossings_n.append(thr.size)
        threshold_crossings_idc.append(thr)
        threshold_crossings_ts.append(thr / sr)

    # Calculate maximum firing frequency
    n_spikes = np.array(threshold_crossings_n)
    max_freq_idx = n_spikes.argmax()
    start_idx = int((step_start * sr).magnitude)
    stop_idx = int((step_stop * sr).magnitude)
    avg_len = int((0.1 * pq.s * sr).magnitude)
    max_freq = n_spikes[max_freq_idx]
    feature_dict["Max. Freq. (Hz)"] = max_freq
    max_freq_volt_trace = block_active.segments[max_freq_idx].analogsignals[0]

    # Calculate the rheobase
    current_amps = np.arange(0, 600, current_delta)
    rheobase_idx = np.where(n_spikes > 0)[0][0]
    feature_dict["Rheobase idx"] = rheobase_idx
    rheobase = current_amps[rheobase_idx]
    feature_dict["Rheobase (pA)"] = rheobase

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

    '''Spike waveform extraction'''
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

    '''Single spike features'''
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
        spikes_dict[spikes_label[idx]]["thr"] = spike_threshold.magnitude - ljp
        # Fast AHP amplitude (from threshold)
        minimum = s.min()
        fast_ahp_amp = minimum - spike_threshold
        spikes_dict[spikes_label[idx]]["fast_ahp"] = fast_ahp_amp
        # Max spike slope in mV/ms
        spike_max_slope = smooth_first_order_gradient.max()
        spikes_dict[spikes_label[idx]]["max_slope"] = spike_max_slope
        # Min Spike slope in mV/ms
        spike_min_slope = smooth_first_order_gradient.min()
        spikes_dict[spikes_label[idx]]["min_slope"] = spike_min_slope
        # AP peak (overshoot)
        spike_peak = s.max()
        spikes_dict[spikes_label[idx]]["peak"] = spike_peak.magnitude - ljp
        # AP half-width (between threshold and peak)
        half_height = ((spike_peak - spike_threshold) / 2.0) + spike_threshold
        left_of_peak = s[: np.argmax(curr_spike)]
        right_of_peak = s[np.argmax(curr_spike) :]
        left_idx = np.abs(left_of_peak - half_height).argmin()
        right_idx = np.abs(right_of_peak - half_height).argmin()
        time_left = left_of_peak.times[left_idx]
        time_right = right_of_peak.times[right_idx]
        half_width = time_right.rescale(pq.ms) - time_left.rescale(pq.ms)
        spikes_dict[spikes_label[idx]]["half_width"] = half_width.magnitude

    # Calculate slow after hyperpolarization
    baseline = max_freq_volt_trace[start_idx - avg_len : start_idx].mean()
    after_step = max_freq_volt_trace[stop_idx : stop_idx + avg_len].mean()
    slow_ahp_amp = after_step - baseline
    feature_dict["Slow AHP (mV)"] = slow_ahp_amp
    # Calculate max freq current
    max_freq_current = current_amps[max_freq_idx]
    feature_dict["I at Max. Freq. (pA)"] = max_freq_current
    # Adaptation Ratio
    isis = np.diff(threshold_crossings_ts[max_freq_idx])
    adaptation_ratio = (isis[-1] / isis[0]).magnitude
    feature_dict["Adaptation ratio"] = adaptation_ratio
    # Average spike time
    avg_spike_time = threshold_crossings_ts[max_freq_idx].mean()
    avg_spike_time = (avg_spike_time - step_start).magnitude
    feature_dict["Avg Spike Time (s)"] = avg_spike_time

    '''Passive Properties'''
    fits = []
    neg_peaks = []
    steady_states = []
    baselines = []
    # fit_inits = (0.025, -70)
    path_passive = list(filter(lambda k: (x in k), passive_abf_files))[0]
    abff =  AxonIO(path_passive)
    block_passive = abff.read_block(signal_group_mode="split-all")
    for sweep in block_passive.segments[1:]:
        bl = sweep.analogsignals[0][stim_start - avg_len : stim_start].mean()
        baselines.append(bl)
        neg_peak = sweep.analogsignals[0][stim_start:stim_stop].min()
        neg_peaks.append(neg_peak)
        neg_peak_idx = sweep.analogsignals[0][stim_start:stim_stop].argmin()
        if neg_peak_idx > 5000:
            neg_peak_idx = 5000
        steady_state = sweep.analogsignals[0][
            stim_stop - avg_len : stim_stop
        ].mean()
        steady_states.append(steady_state)
        decay = sweep.analogsignals[0][
            stim_start + 50 : stim_start + neg_peak_idx
        ]
        decay = decay - decay.min()
        xdata = np.arange(len(decay)) * (1 / sr.magnitude)
        ydata = np.array(decay)[:, 0]
        fit = scipy.optimize.curve_fit(exp_decay,
                                       xdata,
                                       ydata,
                                       p0=(0.015, ydata[0]))
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
    feature_dict["Sag Amplitude (mV)"] = sag_amplitude
    sag_trace = block_passive.segments[sag_idx].analogsignals[0]
    sag_current_trace = block_passive.segments[sag_idx].analogsignals[1]

    # Ohms law: R = V/I
    line_fit = scipy.optimize.curve_fit(line, current_steps, volt_changes)
    input_resistance = line_fit[0][0] * 1000  # *1000 converts to MegaOhm
    feature_dict["Input R (MOhm)"] = input_resistance
    capacitance = (decay_tau / input_resistance) * 1000  # in microfarads
    feature_dict["Capacitance (pF)"] = capacitance

    # Find the resting membrane potential from a list
    resting_potential = annotations.loc[x]["resting potential"]
    feature_dict["Resting (mV)"] = resting_potential

    feature_dict["RS AHP Amp. (mV)"] = spikes_dict["rs"]["fast_ahp"].magnitude
    feature_dict["RS Max. Slope (mV/ms)"] = spikes_dict["rs"]["max_slope"]
    feature_dict["RS Min. Slope (mV/ms)"] = spikes_dict["rs"]["min_slope"]
    feature_dict["RS Peak (mV)"] = spikes_dict["rs"]["peak"]
    feature_dict["RS Half Width (ms)"] = spikes_dict["rs"]["half_width"]
    feature_dict["RS Threshold (mV)"] = spikes_dict["rs"]["thr"]
    feature_dict["LS AHP Amp. (mV)"] = spikes_dict["ls"]["fast_ahp"].magnitude
    feature_dict["LS Max. Slope (mV/ms)"] = spikes_dict["ls"]["max_slope"]
    feature_dict["LS Min. Slope (mV/ms)"] = spikes_dict["ls"]["min_slope"]
    feature_dict["LS Peak (mV)"] = spikes_dict["ls"]["peak"]
    feature_dict["LS Half Width (ms)"] = spikes_dict["ls"]["half_width"]
    feature_dict["LS Threshold (mV)"] = spikes_dict["ls"]["thr"]

    curr_df = pd.DataFrame(feature_dict, index=[x])
    result_df = result_df.append(curr_df)

    trace_dict[x]["sag_trace"] = np.array(sag_trace)[:, 0]
    trace_dict[x]["sag_current_trace"] = np.array(sag_current_trace)[:, 0]
    trace_dict[x]["rheo_trace"] = np.array(
        block_active.segments[rheobase_idx].analogsignals[0]
    )[:, 0]
    trace_dict[x]["rheo_current_trace"] = np.array(
        block_active.segments[rheobase_idx].analogsignals[1]
    )[:, 0]
    trace_dict[x]["max_freq_trace"] = np.array(
        block_active.segments[max_freq_idx].analogsignals[0]
    )[:, 0]
    trace_dict[x]["max_freq_current_trace"] = np.array(
        block_active.segments[max_freq_idx].analogsignals[1]
    )[:, 0]
    trace_dict[x]["time"] = np.arange(len(sag_trace)) * si
    trace_dict[x]["spikes_dict"] = spikes_dict

result_df = result_df.astype(float)

'''Plot example traces'''
# fig = plt.figure(figsize=(8.25, 11.7083333333333334),
#                 constrained_layout=True)
fig = plt.figure(figsize=(7.5, 13.33), constrained_layout=True)
plt.rcParams["svg.fonttype"] = "none"
sns.set(
    context="paper",
    style="ticks",
    palette="colorblind",
    font="Arial",
    font_scale=1,
    color_codes=True,
)
mpl.rcParams.update({"font.size": 12})

gs = fig.add_gridspec(5, 6)
ax1 = fig.add_subplot(gs[0:4, 0:2])
ax2 = fig.add_subplot(gs[0:4, 2:4])
ax3 = fig.add_subplot(gs[0:4, 4:6])
ax4 = fig.add_subplot(gs[4, 0:2])
ax5 = fig.add_subplot(gs[4, 2:4])
ax6 = fig.add_subplot(gs[4, 4:6])

ylim_current = (-250, 600)

k = "D70"
ax = ax1
start = 2000
t = trace_dict[k]["time"] - start
ax.plot(t, np.array(trace_dict[k]["sag_trace"]) - 30, alpha=1, color="k")
ax.plot(t, trace_dict[k]["max_freq_trace"] + 100, alpha=1, color="k")
ax.plot(t, trace_dict[k]["rheo_trace"], alpha=1, color="k")
ax.set_xlim((0, 3500 - start))
ax.set_title("Sst-EYFP")
ax.set_ylim((-135, 165))
ax.set_ylabel("Voltage (mV)")
ax.set_yticks((-100, -50, 0, 50))

ax = ax4
ax.plot(t, np.array(trace_dict[k]["sag_current_trace"]), alpha=1, color="k")
ax.plot(t, trace_dict[k]["max_freq_current_trace"], alpha=1, color="k")
ax.plot(t, trace_dict[k]["rheo_current_trace"], alpha=1, color="k")
ax.set_xlim((0, 3500 - start))
ax.set_ylim(ylim_current)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Current (pA)")
ax.set_yticks(np.arange(-200, 601, 200))

k = "M61"
ax = ax2
t = trace_dict[k]["time"] - start
ax.plot(t, np.array(trace_dict[k]["sag_trace"]) - 30, alpha=1, color="k")
ax.plot(t, trace_dict[k]["max_freq_trace"] + 100, alpha=1, color="k")
ax.plot(t, trace_dict[k]["rheo_trace"], alpha=1, color="k")
ax.set_xlim((0, 3500 - start))
ax.set_title("Slc17a8-EYFP")
ax.set_ylim((-135, 165))
ax.set_yticks((-100, -50, 0, 50))

ax = ax5
ax.plot(t, np.array(trace_dict[k]["sag_current_trace"]), alpha=1, color="k")
ax.plot(t, trace_dict[k]["max_freq_current_trace"], alpha=1, color="k")
ax.plot(t, trace_dict[k]["rheo_current_trace"], alpha=1, color="k")
ax.set_xlim((0, 3500 - start))
ax.set_ylim(ylim_current)
ax.set_xlabel("Time (ms)")
ax.set_yticks(np.arange(-200, 601, 200))

k = "D49"
ax = ax3
t = trace_dict[k]["time"] - start
ax.plot(t, np.array(trace_dict[k]["sag_trace"]) - 30, alpha=1, color="k")
ax.plot(t, trace_dict[k]["max_freq_trace"] + 100, alpha=1, color="k")
ax.plot(t, trace_dict[k]["rheo_trace"], alpha=1, color="k")
ax.set_xlim((0, 3500 - start))
ax.set_title("Putative PC")
ax.set_ylim((-135, 165))
ax.set_yticks((-100, -50, 0, 50))

ax = ax6
ax.plot(t, np.array(trace_dict[k]["sag_current_trace"]), alpha=1, color="k")
ax.plot(t, trace_dict[k]["max_freq_current_trace"], alpha=1, color="k")
ax.plot(t, trace_dict[k]["rheo_current_trace"], alpha=1, color="k")
ax.set_xlim((0, 3500 - start))
ax.set_ylim(ylim_current)
ax.set_xlabel("Time (ms)")
ax.set_yticks(np.arange(-200, 601, 200))

'''Save ephys_df.csv'''
save_path = os.path.join(dirname, "data", "ephys_features_df.csv")
result_df.to_csv("ephys_features_df.csv")
