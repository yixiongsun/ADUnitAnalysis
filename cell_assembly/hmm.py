import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd

import unit_utils
import os

import scipy.io as sio
from scipy import stats
import pickle

from sklearn.decomposition import PCA
import seaborn as sns
from analysis import files
from hmmlearn import hmm

# TO TRY
# 1. Use naive binned firing rates (z scored)
# 2. Use PCA/ICA state space
# 3. Use only ICA components from cell assembly detection

assemblies = []
new = False

for strain in files:
    z = []
    for k in files[strain]:
        rating_file = k['rating_file']
        directories = k['directories']
        tetrodes = k['tetrodes']
        sessions = k['sessions']

        binsize = 20

        for d in range(0, len(directories)):
            print(strain + " " + sessions[d])

            directory = directories[d]

            units = unit_utils.good_units(os.path.dirname(directory), rating_file, tetrodes)

            # Z-scored bins
            # try bin size of 100
            z_matrix, spike_matrix, bins = unit_utils.z_spike_matrix(directory, units, 100, remove_nan=True)
            #
            ripple_data = sio.loadmat(os.path.join(directory, "ripples.mat"))
            ts = ripple_data['ts'].flatten()
            ripple_windows = unit_utils.merge_ripples(ripple_data['ripple_windows'])
            r_bins, r_start_bins = unit_utils.ripple_bins(bins, ripple_windows, ts)

            z_ripple = z_matrix[:, r_bins]



            # shorten to 1000 sample points
            pca = PCA()
            z_transformed = pca.fit_transform(z_matrix.T)

            z_subset = z_transformed[0:2000,:]

            model = hmm.GaussianHMM(n_components=10, covariance_type="full", n_iter=100)
            model.fit(z_transformed)

            out = model.predict(z_ripple.T)


            break
