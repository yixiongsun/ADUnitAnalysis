import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

import unit_utils
import os

import scipy.io as sio
import seaborn as sns
from matplotlib import cm


directory = 'D:\\TetrodeData\\2022-03-23_09-17-15-p\\Sleep1'


rating_file = 'D:\\TetrodeData\\2022-03-23_09-17-15-p\\ClusterRating_2022-03-23_09-17-15-p.csv'

tetrodes = [1,2,3,4,5]

units = unit_utils.good_units(os.path.dirname(directory), rating_file, tetrodes)

# load ripple data
ripple_lfp = sio.loadmat(os.path.join(directory, "ripple_lfp.mat"))['ripple_lfp'].flatten()
ripple_data = sio.loadmat(os.path.join(directory, "ripples.mat"))
num_ripples = np.shape(ripple_data['ripple_windows'])[1]
ripple_windows = ripple_data['ripple_windows']
ripple_peaks = ripple_data['ripple_peaks'].flatten()
ts = ripple_data['ts'].flatten()

total_duration = (ts[-1] - ts[0]) / 1000000




j = 6

# plot first 6
#plt.figure()
plt.rcParams['font.size'] = '16'
plt.rcParams['axes.linewidth'] = 2
#indices = [14,   84,   86,   87,   95,   97,  108,  109,  115,  123,  125]
#indices = [14,  120,  280,  450,  451,  541,  571,  576,  610,  705,  715]
indices = [6]
for j in range(len(indices)):
    k = indices[j]
    spikes = []
    ind = ripple_windows[:,k]

    # plot another ind
    window = 700
    start = ts[ind[0] - window]
    end = ts[ind[1] + window]

    # duration
    duration = int(ind[1] - ind[0] + 2 * window / 2)


    for i in range(0, len(units)):
        spike_ts = units[i]['TS']
        spikes.append(unit_utils.spikes_in_window(spike_ts, start, end))#.size)


    # calculate firing rate
    #print(spikes)

    # get ripple x and y

    y = ripple_lfp[ind[0] - window:ind[1] + window]
    x = ts[ind[0] - window:ind[1] + window]

    fig,axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]}, sharex='col')


    axs[0].plot(x, y, c='black', linewidth=1.0)
    #plt.axis('off')

    yl = axs[0].get_ylim()
    axs[0].set_xlim(x[0],x[-1])
    axs[0].spines.right.set_visible(False)
    axs[0].spines.top.set_visible(False)
    axs[0].spines.bottom.set_visible(False)
    axs[0].grid(False)
    axs[0].add_patch(patches.Rectangle((ts[ind[0]], yl[0]), ts[ind[1]] - ts[ind[0]], yl[1] - yl[0], facecolor='red', alpha=0.3,
                                       edgecolor='none'))
    ind1 = ripple_windows[:, k-1]
    axs[0].add_patch(
        patches.Rectangle((ts[ind1[0]], yl[0]), ts[ind1[1]+50] - ts[ind1[0]], yl[1] - yl[0], facecolor='red', alpha=0.3,
                          edgecolor='none'))
    axs[0].set_yticks([])
    axs[0].set_xticks([start, ts[ind[0] - window + 1500]], [0,800])

    colors = plt.cm.jet(np.linspace(0,1,len(spikes)))

    axs[1].eventplot(spikes, colors='black')
    #plt.axis('off')

    axs[1].spines.right.set_visible(False)
    axs[1].spines.top.set_visible(False)
    axs[1].grid(False)
    yl = axs[1].set_ylim(0.5, len(units)-0.5)
    axs[1].set_yticks([1, len(units)-1], [1, len(units)])
    #ax.set_ylim(-1, 5)
    axs[1].add_patch(
        patches.Rectangle((ts[ind[0]], yl[0]), ts[ind[1]] - ts[ind[0]], yl[1] - yl[0], facecolor='red', alpha=0.3,
                          edgecolor='none'))

    ind1 = ripple_windows[:, k - 1]
    axs[1].add_patch(
        patches.Rectangle((ts[ind1[0]], yl[0]), ts[ind1[1]+50] - ts[ind1[0]], yl[1] - yl[0], facecolor='red', alpha=0.3,
                          edgecolor='none'))

    #plt.savefig('output/ripple_7_raster.png', dpi=300)"""
    axs[1].tick_params(width=2)
    axs[1].set_ylabel("Neuron #")
    axs[1].set_xlabel("Time (ms)")

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('output/firing_dynamics/ripple_raster.png', dpi=300)


