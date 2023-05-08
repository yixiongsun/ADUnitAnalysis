import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd

import unit_utils
import os

import scipy.io as sio
from scipy import stats
from scipy import signal
import pickle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from sklearn.decomposition import FastICA
import seaborn as sns


files = {
    'WT': [{
        'rating_file': 'D:\\TetrodeData\\2022-03-23_09-17-15-p\\ClusterRating_2022-03-23_09-17-15-p.csv',
        'directories': ['D:\\TetrodeData\\2022-03-23_09-17-15-p\\Sleep1',
                        'D:\\TetrodeData\\2022-03-23_09-17-15-p\\Sleep2'],
        'tetrodes': list(range(1, 6))
    },
    {
        'rating_file': 'D:\\TetrodeData\\2022-03-24_09-28-38-p\\ClusterRating_2022-03-24_09-28-38-p.csv',
        'directories': ['D:\\TetrodeData\\2022-03-24_09-28-38-p\\Sleep1',
                        'D:\\TetrodeData\\2022-03-24_09-28-38-p\\Sleep2',
                        'D:\\TetrodeData\\2022-03-24_09-28-38-p\\Sleep3'],
        'tetrodes': list(range(1, 6))

    }],
    'TG': [{
        'rating_file': 'D:\\TetrodeData\\2022-04-12-09-29-11-p\\ClusterRating_2022-04-12_09-29-11-p.csv',
        'directories': ['D:\\TetrodeData\\2022-04-12-09-29-11-p\\Sleep1',
                        'D:\\TetrodeData\\2022-04-12-09-29-11-p\\Sleep2'],
        'tetrodes': list(range(1, 9))
    },
    {
        'rating_file': 'D:\\TetrodeData\\2022-04-13_09-31-41-p\\ClusterRating_2022-04-13_09-31-41-p.csv',
        'directories': ['D:\\TetrodeData\\2022-04-13_09-31-41-p\\Sleep1',
                        'D:\\TetrodeData\\2022-04-13_09-31-41-p\\Sleep2',
                        'D:\\TetrodeData\\2022-04-13_09-31-41-p\\Sleep3'],
        'tetrodes': list(range(1, 9))
    }]
}


k = files["WT"][0]


rating_file = k['rating_file']
directories = k['directories']
tetrodes = k['tetrodes']

units = unit_utils.good_units(os.path.dirname(directories[0]), rating_file, tetrodes)
pyr_units, int_units = unit_utils.split_unit_types(os.path.join(os.path.dirname(rating_file), 'putative_interneurons.csv'), units)

num_windows = 251
total_ripples = 0




for directory in directories:

    ripple_data = sio.loadmat(os.path.join(directory, "ripples.mat"))
    ts = ripple_data['ts'].flatten()
    ripple_windows = unit_utils.merge_ripples(ripple_data['ripple_windows'])
    ripple_filtered = sio.loadmat(os.path.join(directory, "ripple_filtered.mat"))['ripple_filtered'].flatten()
    ripple_lfp = sio.loadmat(os.path.join(directory, "ripple_lfp.mat"))['ripple_lfp'].flatten()

    num_ripples = np.shape(ripple_windows)[1]


    # select a long ripple
    ripple_phase = ripple_data['ripple_phase'].flatten()[4][0].flatten()

    r_durations = ripple_windows[1,:] - ripple_windows[0,:]


    l_w = 1000
    s_w = 40

    #ind = np.argsort(r_durations)[-8]
    ind = np.argsort(r_durations)[-15]
    y = ripple_lfp[ripple_windows[0,ind] - s_w:ripple_windows[1,ind] + s_w]
    x = ts[ripple_windows[0,ind] - s_w:ripple_windows[1,ind] + s_w]
    y_1 = ripple_filtered[ripple_windows[0,ind] - s_w:ripple_windows[1,ind] + s_w]
    y_2 = ripple_lfp[ripple_windows[0,ind] - l_w:ripple_windows[1,ind] + l_w]
    y_3 = ripple_filtered[ripple_windows[0,ind] - l_w:ripple_windows[1,ind] + l_w]

    x_1 = ts[ripple_windows[0,ind] - l_w:ripple_windows[1,ind] + l_w]

    print(x[-1] - x[0])

    ripple_phase = ripple_phase[ripple_windows[0,ind]-s_w:ripple_windows[1,ind]+s_w]

    ripple_phase_peaks, _ = signal.find_peaks(ripple_phase)

    pk = x[ripple_phase_peaks]
    pk = (pk[1:-1] + pk[0:-2])/2
    start = ts[ripple_windows[0,ind]]
    end = ts[ripple_windows[1,ind]]
    spikes = []

    for i in range(0, len(int_units)):
        spike_ts = int_units[i]['TS']
        #if int_units[i]['tetrode'] == 5:

        spikes.append(unit_utils.spikes_in_window(spike_ts, start, end))

    fig, axs = plt.subplots(2, 1, figsize=(8, 4), gridspec_kw={'height_ratios': [1, 1]}, sharex='col')
    #y_2 = y_2 + 1700
    y_2 /= 10
    y_3 /= 4
    axs[0].plot(x_1, y_2, c='black', linewidth=1)
    axs[0].set_xlim(x_1[0], x_1[-1])
    yl = axs[0].get_ylim()
    print(yl)
    axs[1].plot(x_1, y_3, c='black', linewidth=1)
    axs[0].axis('off')
    axs[1].axis('off')
    scalebar = AnchoredSizeBar(axs[1].transData,
                               200 * 1000, '100 ms', 'lower left',
                               pad=0.1,
                               size_vertical=1,
                               frameon=False)

    axs[1].add_artist(scalebar)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.savefig(fname='output/phase_locking_example_zoom_out.png', dpi=300)
    #plt.show()

    #fig, axs = plt.subplots(1, figsize=(8, 5))
    fig, axs = plt.subplots(3, 1, figsize=(8, 5), gridspec_kw={'height_ratios': [3, 3 ,1]}, sharex='col')

    #y_mean = np.mean(y)
    #y = y + 1300
    axs[0].plot(x,y, c='black')
    axs[0].set_ylim(yl[0], yl[1])
    axs[0].set_xlim(x[0], x[-1])
    axs[0].axis('off')
    axs[1].plot(x,y_1, c='black')
    axs[1].axis('off')
    axs[2].eventplot(spikes[1], colors='black', lineoffsets=-200,linelengths=80)
    axs[2].axis('off')


    ylim = axs[0].get_ylim()
    axs[0].vlines(pk, ylim[0], ylim[1], color="lightgray", linestyles='dashed',linewidth=0.5)

    ylim = axs[1].get_ylim()
    axs[1].vlines(pk, ylim[0], ylim[1], color="lightgray", linestyles='dashed', linewidth=0.5)

    ylim = axs[2].get_ylim()
    axs[2].vlines(pk, ylim[0], ylim[1], color="lightgray", linestyles='dashed', linewidth=0.5)

    scalebar = AnchoredSizeBar(axs[1].transData,
                               20 * 1000, '10 ms', 'lower left',
                               pad=0.1,
                               size_vertical=1,
                               frameon=False)

    axs[2].add_artist(scalebar)

    plt.subplots_adjust(wspace=0, hspace=0)

    # Hide axes ticks

    plt.tight_layout
    plt.savefig(fname='output/phase_locking_example.png', dpi=300)
    #plt.show()



    # subplots
    #plt.figure(figsize=(8,6))

    print(int_units[1])

    break








#pyr_units_wt.append(pyr_binned_spikes)
#int_units_wt.append(int_binned_spikes)


#data[strain] = (pyr_units_wt, int_units_wt)