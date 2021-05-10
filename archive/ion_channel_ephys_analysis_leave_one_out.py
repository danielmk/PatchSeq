# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 20:46:53 2020

@author: Daniel
"""

import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
#from pyglmnet import GLM, simulate_glm
from sklearn import linear_model, model_selection

plt.rcParams['svg.fonttype'] = 'none'
sns.set(context='paper',
        style='whitegrid',
        palette='colorblind',
        font='Arial',
        font_scale=2,
        color_codes=True)

# Load count and alignment data and merge them into one annotated dataframe
adata = sc.read_h5ad(r"E:\Dropbox\Dropbox\01_SST Project_daniel\033_PatchSeq CA3 SO\transcriptomics\Patch-Seq\count_exons_introns_full_named_postqc.h5ad")
full_df = pd.read_csv(r"C:\Users\Daniel\repos\PatchSeq\full_df.csv", index_col=0)
ephys_df = pd.read_csv(r"C:\Users\Daniel\repos\PatchSeq\ephys_df.csv", index_col=0)
adata.var_names_make_unique()
adata.obs_names_make_unique()

adata.obs = pd.concat([adata.obs, full_df], axis=1, sort=False, join='inner')
adata.obs = adata.obs.loc[:, ~adata.obs.columns.duplicated()]

adata = adata[adata.obs_names[adata.obs['ephys'] == 1],:]

sc.pp.log1p(adata, base=2, copy=False)

vg_channels = pd.read_csv(r"E:\Dropbox\Dropbox\01_SST Project_daniel\033_PatchSeq CA3 SO\transcriptomics\Patch-Seq\voltage_gated_ion_channels_list.txt",delimiter='\t')
vg_channels_names = [x.title() for x in vg_channels['Approved symbol']]

# Find genes that still exist in adata after quality control
vg_channels_names_exist = list(filter(lambda x: x in adata.var_names, vg_channels_names))

adata_vgcs = adata[:,vg_channels_names_exist]

np.random.seed(354536)
random_names = np.random.choice(adata.var_names, adata_vgcs.shape[1])

adata_randos = adata[:,random_names]

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
    "LS Threshold (mV)"]

model = sm.families.Poisson()
parameter_name = "Max. Freq. (Hz)"

classes = adata.obs[parameter_name]
features = sm.add_constant(adata_vgcs.to_df())

# GLM to predict with leave-one-out validation
predictions = []
for idx_out in range(features.shape[0]):
    drop_label = classes.index[idx_out]
    print(drop_label)
    features_dropped = features.drop(drop_label)
    classes_dropped = classes.drop(drop_label)
    
    poisson_model = sm.GLM(classes_dropped, features_dropped, family=model)
    poisson_results = poisson_model.fit_regularized()
    
    
    to_predict = np.array(features.loc[drop_label,:])[np.newaxis,:]
    prediction = poisson_results.predict(to_predict)
    predictions.append(prediction[0])


fig, ax = plt.subplots(1)
ax.scatter(classes, predictions)
ax.plot([0, 300], [0, 300])
ax.set_title("Trained on VGCS")
ax.set_xlabel("Actual Max Freq.")
ax.set_ylabel("Predicted Max Freq.")

"""
# GLM model on random genes as control
poisson_model_rnd = sm.GLM(y_train_rnd, X_train_rnd, family=sm.families.Poisson())
poisson_results_rnd = poisson_model_rnd.fit()

poisson_train_rnd_prediction = poisson_results_rnd.predict(X_train_rnd)

plt.figure()
plt.scatter(y_train_rnd, poisson_train_rnd_prediction)

poisson_test_rnd_prediction = poisson_results.predict(X_test_rnd)

plt.figure()
plt.scatter(y_test_rnd, poisson_test_rnd_prediction)
"""

#X_train, X_test, y_train, y_test = model_selection.train_test_split(features_vgcs, classes, test_size=1)

"""
# GLM to classify colocalizing cells
features = sm.add_constant(adata_vgcs.to_df())
classes = adata_vgcs.obs['Max. Freq. (Hz)']

poisson_model = sm.GLM(classes, features, family=sm.families.Gaussian())
poisson_results = poisson_model.fit()

print(poisson_results.summary())

poisson_out_df = pd.DataFrame({"classes": classes, "prediction": poisson_results.predict()})
fig, ax = plt.subplots(1)
sns.scatterplot(x="classes", y ="prediction", data=poisson_out_df)

# GLM to classify colocalizing cells
features = sm.add_constant(adata_randos.to_df())
classes = adata_randos.obs['Max. Freq. (Hz)']

poisson_model = sm.GLM(classes, features, family=sm.families.Gaussian())
poisson_results = poisson_model.fit()

print(poisson_results.summary())

poisson_out_df = pd.DataFrame({"classes": classes, "prediction": poisson_results.predict()})
fig, ax = plt.subplots(1)
sns.scatterplot(x="classes", y ="prediction", data=poisson_out_df)
"""