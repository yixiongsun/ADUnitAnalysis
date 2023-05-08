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
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
from scipy import stats
from scipy import signal


import networkx as nx
from itertools import combinations


# examine features of cells part of assemblies
# raster examples


sns.set_style('ticks')

bin_size = '100'

assemblies_100 = pickle.load(open('../output/assemblies_' + bin_size +'.pkl', 'rb'))

bin_size = '20'

assemblies_20 = pickle.load(open('../output/assemblies_' + bin_size +'.pkl', 'rb'))

for strain in assemblies_100:

    for i in range(0, len(assemblies_100[strain])):
        assembly = assemblies_100[strain][i]
        session = assembly['session']

        if session != 'Hab Rec 1' and session != 'OLM Rec 1':
            continue

        post_assembly = assemblies_20[strain][i]

        templates = assembly['templates']
        post_templates = post_assembly['templates']

        x = np.abs(np.inner(templates.T, post_templates.T))
        x = x[:, np.argmax(x, axis=1)]
        plt.figure()
        sns.heatmap(x)
        plt.show()
        # num cells same



        # take next one






