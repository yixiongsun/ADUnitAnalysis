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



assemblies = pickle.load(open('../output/assemblies_pre_template.pkl', 'rb'))

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
    'strength': [],
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



        #n_r_s = np.delete(z_strengths, assembly['ripple_bins'], axis=1)

        activations = unit_utils.reactivation_indices(z_strengths)

        s = []
        # plt.figure()
        for i in range(0, num_templates):
            s.append(np.mean(r_s[i,:]))

        # plt.show()

        for k in s:
            data['strength'].append(k)
            data['strain'].append(strain)
            data['session'].append(session)

        # Try separating assemblies based on corr -> weak vs strong assemblies

df = pd.DataFrame(data)

df = df[df['strain'] == 'WT']

plt.figure()
ax = sns.swarmplot(data=df, x="session", y="strength")

pairs = [
    ("Hab Rec 1", "Hab Rec 2"),
    ("OLM Rec 1", "OLM Rec 2"),
    ("OLM Rec 1", "OLM Rec 3")
]
annot = Annotator(ax, pairs, data=df, x="session", y="strength")
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction='bonferroni',
                text_format='star', loc="outside").apply_test().annotate()
plt.show()


print(df)


