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



assemblies = pickle.load(open('../output/assemblies_pre_template_100.pkl', 'rb'))

reactivation_strengths = []

data = {
    'strength': [],
    'session': [],
    'strain': [],
    'z_strength': [],
    'relative_change': [],
    'corr': [],
    'activation_strength': []
}

for strain in assemblies:
    counter = 1
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

        r_z_matrix = z_matrix[:, assembly['ripple_bins']]


        z_strengths = stats.zscore(strengths, axis=1)
        r_z_strengths = z_strengths[:,assembly['ripple_bins']]


        activations = unit_utils.reactivation_indices(z_strengths)
        # get strengths of activations

        # activations of template 1

        # save template examples
        plt.figure()
        sns.heatmap(templates, cmap='coolwarm')
        plt.savefig('../output/assembly/bin_100/template_' + strain + str(counter) + '.png', dpi=300)
        #plt.show()

        """plt.figure()
        sns.heatmap(np.sum(z_matrix[:, activations[4]], axis=1).reshape(-1,1))
        plt.show(block = False)

        plt.figure()
        sns.heatmap(templates, cmap='coolwarm')
        plt.show()"""

        plt.figure()
        ax1= plt.subplot(211)
        sns.heatmap(z_matrix[:,0:200], cbar=False)

        plt.subplot(212, sharex = ax1)
        for i in range(0, num_templates):
            plt.plot(strengths[i, 0:200])


        #plt.ylim(-3,10)
        plt.savefig('../output/assembly/bin_100/sample_activation_' + strain + str(counter) + '.png', dpi=300)

        counter += 1





