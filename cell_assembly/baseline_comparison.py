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

bin_size = '20'



assemblies = pickle.load(open('../output/assemblies_pre_template_' + bin_size + '.pkl', 'rb'))

reactivation_strengths = []

data = {
    'strength': [],
    'session': [],
    'strain': []
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
        s = []
        # plt.figure()
        for i in range(0, num_templates):
            s.append(np.mean(strengths[i,:]))

        # plt.show()

        for k in s:
            data['strength'].append(k)
            data['strain'].append(strain)
            data['session'].append(session)

        # Try separating assemblies based on corr -> weak vs strong assemblies

df = pd.DataFrame(data)


plt.figure(figsize=(5,8))
ax = sns.violinplot(data=df, x="strain", y="strength")

pairs = [
    ("WT", "TG")
]
annot = Annotator(ax, pairs, data=df, x="strain", y="strength")
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction='bonferroni',
                text_format='star', loc="outside").apply_test().annotate()
plt.savefig('../output/assembly/bin_' + bin_size + '/baseline_strength.png', dpi=300)
#plt.show()


print(df)


