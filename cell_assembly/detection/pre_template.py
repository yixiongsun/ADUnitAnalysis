import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd

import unit_utils
import os

import scipy.io as sio
from scipy import stats
import pickle
from scipy.ndimage import gaussian_filter1d

from sklearn.decomposition import FastICA
import seaborn as sns
from analysis import files


# cell assembly detection based on Detecting cell assemblies in large neuronal populations, 2013
# 1. bin spike trains into matrix
# 2. find number of assemblies based on PC threshold
# 3. project spike matrix onto significant PC subspace
# 4. ICA to identify components

# Analyses:



# 1. Divide time bins into ripple and non ripple bins
# ripple bins will contain non ripple timepoints
# for each ripple, find bin which start and end is in = bounds

# Compare pre and post exposure: post exposure should have higher ripple modulation of assembly activity


assemblies = {
    'WT': [],
    'TG': []
}

# try a couple of bin sizes
# 10, 20, 50, 100


new = True
if os.path.exists('../../output/assemblies_pre_template_50.pkl') and new == False:
    assemblies = pickle.load(open('../../output/assemblies_pre_template_50.pkl', 'rb'))
else:
    for strain in files:
        for k in files[strain]:
            rating_file = k['rating_file']
            directories = k['directories']
            tetrodes = k['tetrodes']
            sessions = k['sessions']


            # 100 ms bin size
            binsize = 50

            directory = directories[0]
            units = unit_utils.good_units(os.path.dirname(directory), rating_file, tetrodes)
            z_matrix, spike_matrix, bins = unit_utils.z_spike_matrix(directory, units, binsize, remove_nan=True)
            templates = unit_utils.assembly_templates_ica(z_matrix)



            for d in range(0,len(directories)):
                print(strain + " " + sessions[d])

                directory = directories[d]


                units = unit_utils.good_units(os.path.dirname(directory), rating_file, tetrodes)
                unit_types = unit_utils.unit_types(os.path.join(os.path.dirname(rating_file), 'putative_interneurons.csv'), units)

                s = unit_utils.spike_trains(directory, units)

                # Z-scored bins
                # try 20 ms bins
                #_, spike_train, _ = unit_utils.z_spike_matrix(directory, units, 10, remove_nan=True)

                #smoothed_spike_train = gaussian_filter1d(spike_train.astype(float), 5, axis=1)
                z_matrix, spike_matrix, bins = unit_utils.z_spike_matrix(directory, units, binsize, remove_nan=True)

                ripple_data = sio.loadmat(os.path.join(directory, "ripples.mat"))
                ts = ripple_data['ts'].flatten()
                ripple_windows = unit_utils.merge_ripples(ripple_data['ripple_windows'])

                # loop through ripple windows

                r_bins, r_start_bins = unit_utils.ripple_bins(bins, ripple_windows, ts)


                #z_hab_s1_non_ripple = np.take(z_hab_s1, nr_bins, 1)

                # try gaussian convolution across spike train


                # get reactivation strengths
                strength = unit_utils.assembly_detection(templates, z_matrix)

                # divide matrix to non ripple and ripple timepoints
                """strength_hab_s1_nripple = np.take(strength_hab_s1, nr_bins, 1)

                strength_hab_s1_ripple = np.take(strength_hab_s1, r_bins, 1)"""

                assemblies[strain].append({
                    'reactivation': strength,
                    'binned': z_matrix,
                    'templates': templates,
                    'ripple_bins': r_bins,
                    'session': sessions[d],
                    'strain': strain,
                    'unit_types': unit_types
                })



    pickle.dump(assemblies, open('../../output/assemblies_pre_template_50.pkl', 'wb'))



