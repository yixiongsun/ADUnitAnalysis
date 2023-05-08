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
from scipy.ndimage import gaussian_filter1d


import networkx as nx
from itertools import combinations


# examine features of cells part of assemblies
# 1. label the cells + plot graphs with FR
# 2. firing properties (firing rate)
# 3. phase locking to ripples


sns.set_style('ticks')


assemblies = pickle.load(open('../output/assemblies_pre_template_20.pkl', 'rb'))

for strain in assemblies:

    for assembly in assemblies[strain]:
        session = assembly['session']
        if session == 'OLM Rec 3':
            continue


        # session type
        session_num = int(session[-1])




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

        activations = unit_utils.reactivation_indices(strengths)

        smoothed_activations = []
        for i in range(0, num_templates):
            activation_signal = np.zeros(z_strengths.shape[1])
            activation_signal[activations[i]] = 1
            smoothed_activations.append(activation_signal)

        smoothed_activations = np.array(smoothed_activations)
        smoothed_activations = gaussian_filter1d(smoothed_activations.astype(float), 1000, axis=1)


        # bin size 20 ms
        # average of 50 windows for 1 second


        # try calculating over time windows, try first 10 minutes = 600 seconds
        for i in range(0, num_templates):

            avg_strengths = []
            for j in range(0, 600):

                avg_strengths.append(np.mean(strengths[i,j*50:(j+1)*50]))

            plt.plot(avg_strengths)


        # calculate activation strengths over time
        # 1. map out over repeated activation
        # 2. map out over fixed time bins
        # 3. map out over ripples
        repeated_activations = []
        for i in range(0, num_templates):
            activation_signal = np.zeros(strengths.shape[1])
            activation_signal[activations[i]] = 1
            r = strengths[i, activations[i]]
            plt.scatter()

            # plot over time

        #
        #repeated_activations =



        if session_num == 1:
            plt.figure()

            plt.subplot(221)
            plt.title(session)

            for i in range(0, num_templates):
                plt.plot(smoothed_activations[i,:])

            plt.ylim(0.0, 0.05)
            # plot raster of assembly
            plt.subplot(222)
            plt.eventplot(activations)
        else:
            plt.subplot(223)
            plt.title(session)

            for i in range(0, num_templates):
                plt.plot(smoothed_activations[i, :])
            plt.ylim(0.0, 0.05)
            # plot raster of assembly
            plt.subplot(224)
            plt.eventplot(activations)
            plt.show()


