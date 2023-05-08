import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd

import unit_utils
import os

import scipy.io as sio
from scipy import stats
import pickle
import matplotlib as mpl

import seaborn as sns
from analysis import files
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
from scipy import stats
from scipy import signal


import networkx as nx
from itertools import combinations


# examine features of cells part of assemblies
# 1. label the cells + plot graphs with FR
# 2. firing properties (firing rate)
# 3. phase locking to ripples


sns.set_style('ticks')


assemblies = pickle.load(open('../output/assemblies_pre_template.pkl', 'rb'))

for strain in assemblies:

    for assembly in assemblies[strain]:
        session = assembly['session']
        if session != 'Hab Rec 1' and session != 'OLM Rec 1':
            continue


        templates = assembly['templates']
        num_templates = templates.shape[1]
        num_cells = templates.shape[0]
        cells = unit_utils.extract_cell_numbers(templates)
        unit_types = assembly['unit_types']
        #tt = assembly['tt']


        # get template reactivation strengths
        strengths = assembly['reactivation']

        # zscore
        z_strengths = stats.zscore(strengths, axis=1)
        avg_strength = np.mean(strengths, axis=1)

        activations = unit_utils.reactivation_indices(z_strengths)

        activation_signal = np.zeros(z_strengths.shape[1])
        activation_signal[activations[0]] = 1


        # plot heatmaps of ripple centered - both activation strength and activation events
        # for each ripple bin, plot +/- 50 bins
        ripple_centered_strength = []
        for r in assembly['ripple_bins']:
            if r - 50 < 0 or r + 50 >= activation_signal.size:
                continue
            ripple_centered_strength.append(activation_signal[r - 50:r + 50])
        ripple_centered_strength = np.array(ripple_centered_strength)
        sns.heatmap(ripple_centered_strength)








