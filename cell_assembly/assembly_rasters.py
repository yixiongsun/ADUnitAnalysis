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
# raster examples


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

        # get raster plots of activations

