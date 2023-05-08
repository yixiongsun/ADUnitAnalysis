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

bin_size = '20'

assemblies = pickle.load(open('../output/assemblies_pre_template_' + bin_size +'.pkl', 'rb'))

for strain in assemblies:

    for assembly in assemblies[strain]:
        session = assembly['session']
        if session != 'Hab Rec 1' and session != 'OLM Rec 1':
            continue


        templates = assembly['templates']
        num_templates = templates.shape[1]
        num_cells = templates.shape[0]

        # plot stem plots
        cells = unit_utils.extract_cell_numbers(templates)

        plt.figure()



        for j in range(0, num_templates):
            ax = plt.subplot(1,num_templates,j+1)

            _, stemlines, _ = ax.stem(templates[:,j], orientation='horizontal')

            line = ax.get_lines()
            xd = line[0].get_xdata()
            yd = line[0].get_ydata()
            c = list(map(lambda x: 'blue' if x == 0.0 else 'red', cells[:, j]))

                     # mec and mfc stands for markeredgecolor and markerfacecolor
            for i in range(0,num_cells):

                plt.plot([xd[i]], [yd[i]], 'o', ms=7, mfc=c[i], mec=c[i])
                #plt.setp(stemlines[i], 'color', c[i])



        plt.figure()
        sns.heatmap(templates, cmap='coolwarm')
        plt.show(block=False)
        plt.figure()
        sns.heatmap(cells, cmap='coolwarm')
        plt.show()


        # extract cells from assembly
        unit_types = assembly['unit_types']
        #tt = assembly['tt']


