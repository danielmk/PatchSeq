# -*- coding: utf-8 -*-
"""
This script analyses the difference between the transgenically
defined cell types using rank sum tests for each feature. 
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
from scipy.stats import ranksums

"""Data"""
dirname = os.path.dirname(__file__)
data_path = os.path.join(dirname, 'data', 'ephys_full_df.csv')

data_df = pd.read_csv(data_path,
                        delimiter=",",
                        index_col=0)

result_dict = {}
features = ['Max. Freq. (Hz)',
            'Slow AHP (mV)',
            'Rheobase (pA)',
            'I at Max. Freq. (pA)',
            'Adaptation ratio',
            'Avg Spike Time (s)',
            'Input R (MOhm)',
            'Capacitance (pF)',
            'Sag Amplitude (mV)',
            'Resting (mV)',
            'RS AHP Amp. (mV)',
            'RS Max. Slope (mV/ms)',
            'RS Min. Slope (mV/ms)',
            'RS Peak (mV)',
            'RS Half Width (ms)',
            'RS Threshold (mV)',
            'LS AHP Amp. (mV)',
            'LS Max. Slope (mV/ms)',
            'LS Min. Slope (mV/ms)',
            'LS Peak (mV)',
            'LS Half Width (ms)',
            'LS Threshold (mV)']

sst = data_df[data_df.label == 'SST-EYFP']
slc17a8 = data_df[data_df.label == 'VGlut3-EYFP']
for f in features:
    x = sst[f]
    y = slc17a8[f]
    result = ranksums(x, y)
    result_dict[f] = [result.statistic, result.pvalue]

result_df = pd.DataFrame.from_dict(result_dict, 
                                   orient='index',
                                   columns=['Statistic', 'pvalue'])
output_path = os.path.join(dirname, 'data', 'stats_ephys_transgenic_type.csv')
result_df.to_csv(output_path)