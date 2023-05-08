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
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

from elephant.statistics import mean_firing_rate
# Try PCA transformation

assemblies = []
new = False

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

            # Z-scored bins
            z_matrix, spike_matrix, bins = unit_utils.z_spike_matrix(directory, units, binsize, remove_nan=True)

            ripple_data = sio.loadmat(os.path.join(directory, "ripples.mat"))
            ts = ripple_data['ts'].flatten()
            ripple_windows = unit_utils.merge_ripples(ripple_data['ripple_windows'])
            r_bins, r_start_bins = unit_utils.ripple_bins(bins,ripple_windows,ts)


            z_ripple = z_matrix[:,r_bins]


            pca = PCA()
            z_transformed = pca.transform(z_matrix.T)

            # try generating trajectories

            z_ripple_transformed = pca.transform(z_ripple.T)


            # try isomaps
            iso = Isomap(n_neighbors=8)
            iso.fit(z_ripple.T)
            z_manifold = iso.transform(z_ripple.T)


            #t = np.arange(z_manifold.shape[0])
            #plt.scatter(z_manifold[:, 0], z_manifold[:, 1], c=t)
            #z_manifold_all = iso.transform(z_matrix.T)

            plt.scatter(z_transformed[:,0], z_transformed[:,1])

            break
