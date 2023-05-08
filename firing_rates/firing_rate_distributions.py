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

new = False

if os.path.exists('../output/firing_rates_distributions.csv') and new == False:
    df = pd.read_csv('../output/firing_rates_distributions.csv')
else:
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

        },
        {
            'rating_file': 'D:\\TetrodeData\\2022-07-08_10-01-42-p\\ClusterRating_2022-07-08_10-01-42-p.csv',
            'directories': ['D:\\TetrodeData\\2022-07-08_10-01-42-p\\Sleep1',
                            'D:\\TetrodeData\\2022-07-08_10-01-42-p\\Sleep2'],
            'tetrodes': [1, 2, 3, 5, 7, 8]
        },
        {
            'rating_file': 'D:\\TetrodeData\\2022-07-09_09-54-32-p\\ClusterRating_2022-07-09_09-54-32-p.csv',
            'directories': ['D:\\TetrodeData\\2022-07-09_09-54-32-p\\Sleep1',
                            'D:\\TetrodeData\\2022-07-09_09-54-32-p\\Sleep2',
                            'D:\\TetrodeData\\2022-07-09_09-54-32-p\\Sleep3'],
            'tetrodes': [1, 2, 3, 5, 7, 8]
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
        },
        {
            'rating_file': 'D:\\TetrodeData\\2022-07-10_13-27-33-p\\ClusterRating_2022-07-10_13-27-33-p.csv',
            'directories': ['D:\\TetrodeData\\2022-07-10_13-27-33-p\\Sleep1',
                            'D:\\TetrodeData\\2022-07-10_13-27-33-p\\Sleep2'],
            'tetrodes': [2,3,5,6,7,8]
        },
        {
            'rating_file': 'D:\\TetrodeData\\2022-07-11_13-47-33-p\\ClusterRating_2022-07-11_13-47-33-p.csv',
            'directories': ['D:\\TetrodeData\\2022-07-11_13-47-33-p\\Sleep1',
                            'D:\\TetrodeData\\2022-07-11_13-47-33-p\\Sleep2',
                            'D:\\TetrodeData\\2022-07-11_13-47-33-p\\Sleep3'],
            'tetrodes': [2,3,5,6,7,8]
        }
        ]
    }



    firing_rates = {
        'fr': [],
        'strain': [],
        'class': []
    }

    # 1. unit firing rate distributions

    for k in files:

        for j in range(0, len(files[k])):

            rating_file = files[k][j]['rating_file']
            directories = files[k][j]['directories']
            tetrodes = files[k][j]['tetrodes']

            units = unit_utils.good_units(os.path.dirname(directories[0]), rating_file, tetrodes)
            pyr_units, int_units = unit_utils.split_unit_types(os.path.join(os.path.dirname(rating_file), 'putative_interneurons.csv'), units)

            total_spikes_pyr = np.zeros(len(pyr_units))
            total_spikes_int = np.zeros(len(int_units))

            duration = 0

            for directory in directories:

                ripple_data = sio.loadmat(os.path.join(directory, "ripples.mat"))
                ts = ripple_data['ts'].flatten()
                ripple_windows = unit_utils.merge_ripples(ripple_data['ripple_windows'])
                duration += (ts[-1] - ts[0]) / 1000000

                for i in range(0, len(pyr_units)):
                    unit = pyr_units[i]
                    spike_ts = unit['TS']
                    total_spikes_pyr[i] += unit_utils.spikes_in_window(spike_ts, ts[0], ts[-1]).size

                for i in range(0, len(int_units)):
                    unit = int_units[i]
                    spike_ts = unit['TS']
                    total_spikes_int[i] += unit_utils.spikes_in_window(spike_ts, ts[0], ts[-1]).size


            for i in total_spikes_pyr:
                firing_rates['fr'].append(i/duration)
                firing_rates['strain'].append(k)
                firing_rates['class'].append('PYR')

            for i in total_spikes_int:
                firing_rates['fr'].append(i/duration)
                firing_rates['strain'].append(k)
                firing_rates['class'].append('INT')

    df = pandas.DataFrame(firing_rates)
    df.to_csv('output/firing_rates_distributions.csv')


sns.set_style('whitegrid')

"""wt_df = df[df['strain'] == 'WT']
ax = sns.histplot(x='fr', hue='class', data=wt_df)
ax.set(xlabel='Firing rate Hz', ylabel='Count')
plt.title('Distribution of firing rates between INT and PYR units (WT)', y=1.08)
plt.savefig(fname='output/firing_dynamics/firing_rate_distributions_wt.png', bbox_inches='tight', dpi=300)"""

"""tg_df = df[df['strain'] == 'TG']
ax = sns.histplot(x='fr', hue='class', data=tg_df)
ax.set(xlabel='Firing rate Hz', ylabel='Count')
plt.title('Distribution of firing rates between INT and PYR units (TG)', y=1.08)
plt.savefig(fname='output/firing_dynamics/firing_rate_distributions_tg.png', bbox_inches='tight', dpi=300)"""

