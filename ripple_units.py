import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

import unit_utils
import os

import scipy.io as sio
import seaborn as sns
from matplotlib import cm
from sklearn.decomposition import PCA
from scipy import stats
from functools import reduce

directory = 'D:\\TetrodeData\\2022-03-23_09-17-15-p\\Sleep2'


rating_file = 'D:\\TetrodeData\\2022-03-23_09-17-15-p\\ClusterRating_2022-03-23_09-17-15-p.csv'

"""directory = 'D:\\TetrodeData\\2022-03-24_09-28-38-p\\Sleep2'


rating_file = 'D:\\TetrodeData\\2022-03-24_09-28-38-p\\ClusterRating_2022-03-24_09-28-38-p.csv'"""

tetrodes = [1,2,3,4,5]

units = unit_utils.good_units(os.path.dirname(directory), rating_file, tetrodes)
pyr_units, int_units = unit_utils.split_unit_types(os.path.join(os.path.dirname(rating_file), 'putative_interneurons.csv'), units)

# load ripple data
#ripple_lfp = sio.loadmat(os.path.join(directory, "ripple_lfp.mat"))['ripple_lfp'].flatten()
ripple_data = sio.loadmat(os.path.join(directory, "ripples.mat"))
num_ripples = np.shape(ripple_data['ripple_windows'])[1]
ripple_windows = ripple_data['ripple_windows']
ripple_peaks = ripple_data['ripple_peaks'].flatten()
ts = ripple_data['ts'].flatten()

total_duration = (ts[-1] - ts[0]) / 1000000

# cap num ripples to 1500

num_ripples = 1500
# plot first 6
#plt.figure()

all_spikes = []

for j in range(num_ripples):
    spikes = []
    ind = ripple_windows[:,j]

    start = ts[ind[0]]
    end = ts[ind[1]]

    # ripple duration in seconds
    duration = (end - start) * (1000 * ripple_data['sampleRate'].flatten())


    for i in range(0, len(pyr_units)):
        spike_ts = pyr_units[i]['TS']
        spikes.append(unit_utils.spikes_in_window(spike_ts, start, end).size)#.size)"""

    for i in range(0, len(int_units)):
        spike_ts = int_units[i]['TS']
        spikes.append(unit_utils.spikes_in_window(spike_ts, start, end).size)#.size)"""

    s = np.array(spikes)
    s = s/duration
    all_spikes.append(s)

    # calculate firing rate
    #print(spikes)
    pca = PCA()
    # get ripple x and y
   #y = ripple_lfp[ind[0]:ind[1]]
    #x = ts[ind[0]:ind[1]]


#plt.subplots_adjust(wspace=0, hspace=0)

all_spikes = np.array(all_spikes)
z = stats.zscore(all_spikes, nan_policy='raise', axis=0)
z = np.nan_to_num(z)


templates = unit_utils.assembly_templates_ica(z.T)
strengths = unit_utils.assembly_detection(templates, z.T)

r_i = unit_utils.reactivation_indices(strengths)

ripple_assembly_count = []
for i in range(num_ripples):
    count = 0
    for j in r_i:
        if i in j:
            count += 1
    ripple_assembly_count.append(count)


ripple_assembly_count = np.array(ripple_assembly_count)
print(np.count_nonzero(ripple_assembly_count == 0))
print(np.count_nonzero(ripple_assembly_count == 1))
print(np.count_nonzero(ripple_assembly_count == 2))
print(np.count_nonzero(ripple_assembly_count == 3))
print(np.count_nonzero(ripple_assembly_count == 4))


#print(r_i)
#sns.heatmap(templates)
"""plt.plot(strengths[0,:])
plt.plot(strengths[1,:])
plt.plot(strengths[2,:])"""
plt.hist(r_i[0])
plt.show()


# labels = kmeans.predict(X.reshape(-1, 1))


z_transformed = pca.fit_transform(z)
plt.scatter(z_transformed[:, 0], z_transformed[:, 1])
#sns.heatmap(all_spikes.T)
plt.show()

plt.scatter(z_transformed[:, 1], z_transformed[:, 2])
#sns.heatmap(all_spikes.T)
plt.show()

plt.scatter(z_transformed[:, 0], z_transformed[:, 2])
#sns.heatmap(all_spikes.T)
plt.show()

# notes: interneurons fire consistently across ripples
# pyr neurons have specific patterns of firing -> can we find repeated patterns?
# coactivations - possible both assembly patterns part of ripple?
# fine firing does not match, only matches roughly
# examine distribution -> spread around, not equal
# examine number of ripple events with overlapping activations -> most ripple no reactivations, few 1,2 and 3

# OLM PRE
# 1289
# 191
# 15
# 4
# 1
#
# OLM POST
# 1290
# 187
# 23
# 0
# 0
#
#
# OLM PRE
# 1308
# 168
# 21
# 3
# 0
#
# OLM POST
# 1283
# 197
# 19
# 1
# 0

# try different animals -> configure inputs
# also look at reactivation strengths, not just count