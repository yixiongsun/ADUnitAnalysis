import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd

import unit_utils
import os
import quantities as pq
import scipy.io as sio
from scipy import stats
import pickle

from sklearn.decomposition import FastICA
import seaborn as sns
from analysis import files
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from elephant.gpfa import GPFA

from elephant.statistics import mean_firing_rate
# Try PCA transformation

assemblies = []
new = False

# need to segregate by trial? ripple centered spike trains

for strain in files:
    for k in files[strain]:
        rating_file = k['rating_file']
        directories = k['directories']
        tetrodes = k['tetrodes']
        sessions = k['sessions']

        binsize = 20

        # use pandas to store data

        for d in range(0, len(directories)):
            print(strain + " " + sessions[d])

            directory = directories[d]

            units = unit_utils.good_units(os.path.dirname(directory), rating_file, tetrodes)

            spike_trains = unit_utils.ripple_spike_trains(directory, units)

            # 100 ms bin size
            bin_size = 100 * pq.ms
            latent_dimensionality = 2

            gpfa_2dim = GPFA(bin_size=bin_size, x_dim=latent_dimensionality)

            # try fitting on 10000 data points
            traj = gpfa_2dim.fit_transform(spike_trains)

            break
