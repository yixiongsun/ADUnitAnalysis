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

bin_size = '100'


assemblies = pickle.load(open('../output/assemblies_pre_template_' + bin_size + '.pkl', 'rb'))

reactivation_strengths = []

data = {
    'strength': [],
    'session': [],
    'strain': [],
    'z_strength': [],
    'relative_change': [],
    'corr': [],
    'delta_corr': [],
    'activation_strength': []
}

for strain in assemblies:
    for assembly in assemblies[strain]:
        session = assembly['session']

        # only use rec 1
        if session != 'Hab Rec 1' and session != 'OLM Rec 1':
            continue

        z_matrix = assembly['binned']
        templates = assembly['templates']
        num_templates = templates.shape[1]

        strengths = assembly['reactivation']
        mean_s = np.mean(strengths, axis=1)



        z_strengths = stats.zscore(strengths, axis=1)

        r_s = np.mean(strengths[:, assembly['ripple_bins']], axis=1)
        r_i = (r_s - mean_s)/mean_s
        r_z = np.mean(z_strengths[:, assembly['ripple_bins']], axis=1)

        activations = unit_utils.reactivation_indices(z_strengths)
        # get strengths of activations

        a_s = []
        for k in range(0, num_templates):
            a_s.append(np.mean(z_strengths[k,activations[k]]))

        c, l, max_corr, diff = unit_utils.activation_correlation(num_templates, z_strengths, activations, assembly['ripple_bins'])


        plt.figure()
        for k in range(0, num_templates):
            midpoint = int(l[k].size/2)
            plt.plot(l[k][midpoint - 50:midpoint + 50], c[k][midpoint - 50:midpoint + 50])

        # plt.show()

        for k in range(0, num_templates):
            data['strength'].append(r_s[k])
            data['strain'].append(strain)
            data['session'].append(session)
            data['z_strength'].append(r_z[k])
            data['relative_change'].append(r_i[k])
            data['corr'].append(max_corr[k])
            data['activation_strength'].append(a_s[k])
            data['delta_corr'].append(diff[k])

        # Try separating assemblies based on corr -> weak vs strong assemblies

df = pd.DataFrame(data)

"""
STRENGTH
"""
plt.figure()
ax = sns.violinplot(data=df, x="strain", y="strength")

pairs = [
    ("WT", "TG")
]
annot = Annotator(ax, pairs, data=df, x="strain", y="strength")
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction='bonferroni',
                text_format='star', loc="outside").apply_test().annotate()
plt.savefig('../output/assembly/bin_' + bin_size + '/ripple_strength.png', dpi=300)



"""
RELATIVE STRENGTH
"""
plt.figure()
ax = sns.violinplot(data=df, x="strain", y="relative_change")

pairs = [
    ("WT", "TG")
]
annot = Annotator(ax, pairs, data=df, x="strain", y="relative_change")
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction='bonferroni',
                text_format='star', loc="outside").apply_test().annotate()
plt.savefig('../output/assembly/bin_' + bin_size + '/ripple_relative_change.png', dpi=300)



"""
Z STRENGTH
"""
plt.figure()
ax = sns.violinplot(data=df, x="strain", y="z_strength")

pairs = [
    ("WT", "TG")
]
annot = Annotator(ax, pairs, data=df, x="strain", y="z_strength")
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction='bonferroni',
                text_format='star', loc="outside").apply_test().annotate()
plt.savefig('../output/assembly/bin_' + bin_size + '/ripple_z_strength.png', dpi=300)



