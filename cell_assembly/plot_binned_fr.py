import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd

import unit_utils
import os

import scipy.io as sio
from scipy import stats
import pickle

from sklearn.decomposition import FastICA
import seaborn as sns
from analysis import files
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
from scipy import stats
from scipy import signal

sns.set_style('ticks')



assemblies = pickle.load(open('../output/assemblies_post_template.pkl', 'rb'))

reactivation_strengths = []

assembly_correlations = {
    'WT': {
        'Hab Rec 1': [],
        'Hab Rec 2': [],
        'OLM Rec 1': [],
        'OLM Rec 2': [],
        'OLM Rec 3': [],
    },
    'TG': {
        'Hab Rec 1': [],
        'Hab Rec 2': [],
        'OLM Rec 1': [],
        'OLM Rec 2': [],
        'OLM Rec 3': [],
    }
}


data = {
    'corr': [],
    'session': [],
    'strain': []
}

for strain in assemblies:
    for assembly in assemblies[strain]:
        session = assembly['session']
        z_matrix = assembly['binned']
        templates = assembly['templates']
        num_templates = templates.shape[1]
        r_z_matrix = z_matrix[:, assembly['ripple_bins']]
        n_r_z_matrix = np.delete(z_matrix, assembly['ripple_bins'], axis=1)

        # plt.figure()
        # sns.heatmap(n_r_z_matrix[:,0:200])
        # plt.show(block = False)

        # plt.figure()
        # sns.heatmap(r_z_matrix[:,0:200])
        # plt.show(block = False)

        strengths = assembly['reactivation']

        # zscore
        z_strengths = stats.zscore(strengths, axis=1)

        r_s = z_strengths[:, assembly['ripple_bins']]

        n_r_s = np.delete(z_strengths, assembly['ripple_bins'], axis=1)

        activations = unit_utils.reactivation_indices(z_strengths)

        max_corr = []
        # plt.figure()
        for i in range(0, num_templates):
            activation_signal = np.zeros(z_strengths.shape[1])
            activation_signal[activations[i]] = 1
            ripple_signal = np.zeros(z_strengths.shape[1])
            ripple_signal[assembly['ripple_bins']] = 1

            activation_signal = activation_signal / np.linalg.norm(activation_signal)
            ripple_signal = ripple_signal / np.linalg.norm(ripple_signal)

            # try correlating with strengths
            # z_signal = z_strengths[0,:] / np.linalg.norm(z_strengths[0,:])

            corr = signal.correlate(activation_signal, ripple_signal)
            lags = signal.correlation_lags(len(activation_signal), len(ripple_signal))

            min = int(lags.size / 2) - 30
            max = int(lags.size / 2) + 30
            # plt.plot(lags[min:max], corr[min:max])

            max_corr.append(np.max(corr))

        print(max_corr)
        # plt.show()

        for k in max_corr:
            data['corr'].append(k)
            data['strain'].append(strain)
            data['session'].append(session)

        # Try separating assemblies based on corr -> weak vs strong assemblies

df = pd.DataFrame(data)

#df = df[df['strain'] == 'TG']

plt.figure()
ax = sns.swarmplot(data=df, x="session", y="corr", hue="strain")

"""pairs = [
    ("Hab Rec 1", "Hab Rec 2"),
    ("OLM Rec 1", "OLM Rec 2"),
    ("OLM Rec 1", "OLM Rec 3")
]
annot = Annotator(ax, pairs, data=df, x="session", y="corr")
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction='bonferroni',
                text_format='star', loc="outside").apply_test().annotate()"""
plt.show()


print(df)

    # bin size 100 ms
    # plot cross correlation only for +/- 1 second


    # normalize cross correlations
    # calculate for each template

    #r = unit_utils.reactivation_indices(r_s)
    #n = unit_utils.reactivation_indices(n_r_s)
    # means look fine, overall increase

    # compare between TG and WT for baseline

    # look at total number of cells participating in the assembly?


