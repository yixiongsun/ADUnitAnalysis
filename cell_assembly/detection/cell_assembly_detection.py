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


new = True
if os.path.exists('../../output/assemblies_50.pkl') and new == False:
    assemblies = pickle.load(open('../../output/assemblies_50.pkl', 'rb'))
else:
    for strain in files:
        for k in files[strain]:
            rating_file = k['rating_file']
            directories = k['directories']
            tetrodes = k['tetrodes']
            sessions = k['sessions']


            # 100 ms bin size
            binsize = 50


            # use pandas to store data


            for d in range(0,len(directories)):
                print(strain + " " + sessions[d])

                directory = directories[d]


                units = unit_utils.good_units(os.path.dirname(directory), rating_file, tetrodes)
                unit_types = unit_utils.unit_types(os.path.join(os.path.dirname(rating_file), 'putative_interneurons.csv'), units)
                tt = [u['tetrode'] for u in units]

                s = unit_utils.spike_trains(directory, units)

                # Z-scored bins
                # try 20 ms bins
                #_, spike_train, _ = unit_utils.z_spike_matrix(directory, units, 10, remove_nan=True)

                #smoothed_spike_train = gaussian_filter1d(spike_train.astype(float), 5, axis=1)
                z_hab_s1, spike_matrix, bins = unit_utils.z_spike_matrix(directory, units, binsize, remove_nan=True)

                ripple_data = sio.loadmat(os.path.join(directory, "ripples.mat"))
                ts = ripple_data['ts'].flatten()
                ripple_windows = unit_utils.merge_ripples(ripple_data['ripple_windows'])

                # loop through ripple windows

                r_bins, r_start_bins = unit_utils.ripple_bins(bins, ripple_windows, ts)

                # find templates
                templates = unit_utils.assembly_templates_ica(z_hab_s1)

                #z_hab_s1_non_ripple = np.take(z_hab_s1, nr_bins, 1)

                # try gaussian convolution across spike train


                # get reactivation strengths
                strength_hab_s1 = unit_utils.assembly_detection(templates, z_hab_s1)

                # divide matrix to non ripple and ripple timepoints
                """strength_hab_s1_nripple = np.take(strength_hab_s1, nr_bins, 1)

                strength_hab_s1_ripple = np.take(strength_hab_s1, r_bins, 1)"""

                assemblies[strain].append({
                    'reactivation': strength_hab_s1,
                    'binned': z_hab_s1,
                    'templates': templates,
                    'ripple_bins': r_bins,
                    'session': sessions[d],
                    'strain': strain,
                    'unit_types': unit_types,
                    'tt': tt
                })



    pickle.dump(assemblies, open('../../output/assemblies_50.pkl', 'wb'))



# convert to pd
# plot 1 assembly strength
"""assembly = assemblies[0]
print(np.mean(assembly["ripple_assembly_strengths"], axis=1))
print(np.mean(assembly["non_ripple_assembly_strengths"], axis=1))
print(assembly["templates"].shape[1])"""



# find match based on ripple times

# zcore strength
#z_strength_hab_s1 = stats.zscore(strength_hab_s1, 1)

# find ripple windows





# relative increase does not change after exposure to OLM
# assembly detection higher in WT than transgenic animals
# Try decreasing time window?
# 1. Separate assembly patterns by those that reactivate strongly during ripples: are they modulated more after exposure? Do the same for weakly reactivated assemblies
# 2. Separate ripples by assembly pattern reactivation strengths -> high reactivation ripples vs low reactivation ripples
# -> analyze distribution of reactivation strength in high activity ripples
# Analyze unit composition: interneuron and pyramidal neurons


#plt.plot(strength_hab_s1_ripple.T)
#plt.show()

# plot raster aligned with first bin
# each row in raster = template
# color = avg assembly strength at that ripple time
# aligned at ripple bin onset, window size +/- 10 bins

"""num_templates = templates.shape[1]


    # 400 ms before and after ripple
    assembly_strengths = np.zeros((num_templates, 41, num_ripples))
    for j in range(0, len(r_start_bins)):
        r_ind = r_start_bins[j]

        for i in range(r_ind - 20, r_ind + 21):
            assembly_strengths[:, i - r_ind + 20, j] = strength_hab_s1[:, i]

    mean_strengths = np.mean(assembly_strengths, 2)

    sns.heatmap(mean_strengths)"""




# mean_z = stats.zscore(mean_strengths, 1)

# compare mean activation strength during and outside ripples
# zscore strengths

# plot cdf
# ks score statistic as difference
# look at mean differences -> fold change



# heatmap plot


# 1. Use Post sleep2/3 as template and Pre as match, if reactivation large in pre -> suggests minimal learning
# 2. Examine cell assembly detection strength during and outside of SWRs for each condition, using non SWR times as templates
# 3. Redo analysis of 1 using SWR only time bins






# Pre sleep
# SWR bins defined as bins starting at SWR onset until end, inclusive
#z_hab_s1, _ = unit_utils.z_spike_matrix(directories[0], units, 20)
#templates = unit_utils.assembly_templates_ica(z_hab_s1)
#strength_hab_s1 = unit_utils.assembly_detection(templates, z_hab_s1)

# plot raster of strength at ripple onset
# need to align bins starting at ripple onset
# get ripple onset bin indices
"""ripple_data = sio.loadmat(os.path.join(directory, "ripples.mat"))
ripple_windows = ripple_data['ripple_windows']
start = ts[0]
stop = ts[-1]

# binsize in ms
binsize = binsize * 1000

# bins until end of timepoint -> exclude last


bins = np.arange(start, stop, binsize)
bin_ind = []
for j in range(0, np.shape(ripple_windows)[1]):

    for i in range(0, bins.size):
        if bins[i] > ripple_windows[0, j]:
            bin_ind.append(i-1)
            break


# select time window of +/- 10 bins
for i in bin_ind:
    strength_hab_s1[,i-10]"""

"""

# Hab
z_hab_s1, _ = unit_utils.z_spike_matrix(directories[0], units, 100)
z_hab_s2, _ = unit_utils.z_spike_matrix(directories[1], units, 100)

# calculate template based on sleep 2
templates = unit_utils.assembly_templates_ica(z_hab_s2)

# detect assemblies in s1 and s2
strength_hab_s1 = unit_utils.assembly_detection(templates, z_hab_s1)
strength_hab_s2 = unit_utils.assembly_detection(templates, z_hab_s2)

# calculate average and subtract between assemblies

print(np.mean(strength_hab_s2, 1) - np.mean(strength_hab_s1, 1))
print(np.mean(np.mean(strength_hab_s2, 1) - np.mean(strength_hab_s1, 1)))


# Train
units = unit_utils.good_units(os.path.dirname(directories[2]), rating_files[1], tetrodes)

# fix nan
z_olm_s1, keep1 = unit_utils.z_spike_matrix(directories[2], units, 100)
z_olm_s2, keep2 = unit_utils.z_spike_matrix(directories[3], units, 100)

indices = np.logical_and(keep1, keep2)

print(indices.sum())
z_olm_s1 = z_olm_s1[indices]
z_olm_s2 = z_olm_s2[indices]




# calculate template based on sleep 2
templates = unit_utils.assembly_templates_ica(z_olm_s2)

# detect assemblies in s1 and s2
strength_olm_s1 = unit_utils.assembly_detection(templates, z_olm_s1)
strength_olm_s2 = unit_utils.assembly_detection(templates, z_olm_s2)

# calculate average and subtract between assemblies

print(np.mean(strength_olm_s2, 1) - np.mean(strength_olm_s1, 1))
print(np.mean(np.mean(strength_olm_s2, 1) - np.mean(strength_olm_s1, 1)))
"""