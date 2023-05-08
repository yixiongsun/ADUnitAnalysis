import pandas as pd
import pycircstat
import matplotlib.pyplot as plt
import numpy as np

import unit_utils
import os
import seaborn as sns
import scipy.io as sio
import pickle
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest

from analysis import files


phase_info = {
    'session': [],
    'p_value': [],
    'z_value': [],
    'strain': [],
    'sig': [],
    'mean': [],
    'class': []
}

bins = np.linspace(-np.pi, np.pi, 33)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = bins[1:] - bins[:-1]


for strain in files:
    for k in files[strain]:
        rating_file = k['rating_file']
        directories = k['directories']
        tetrodes = k['tetrodes']
        sessions = k['sessions']
        if not k['spindles']:
            continue

        units = unit_utils.good_units(os.path.dirname(directories[0]), rating_file, tetrodes)
        pyr_units, int_units = unit_utils.split_unit_types(os.path.join(os.path.dirname(rating_file), 'putative_interneurons.csv'), units)


        for d in range(0, len(directories)):
            directory = directories[d]
            # Get ripple and spindle windows
            # extract only ripple events occurring during a spindle

            ripple_data = sio.loadmat(os.path.join(directory, "ripples.mat"))
            ripple_phases = ripple_data['ripple_phase'].flatten()
            ripple_peaks = ripple_data['ripple_peaks'].flatten()

            ripple_windows = ripple_data['ripple_windows']
            ts = ripple_data['ts'].flatten()
            num_ripples = np.shape(ripple_data['ripple_windows'])[1]

            spindle_data = sio.loadmat(os.path.join(directory, "spindles.mat"))
            spindle_windows = spindle_data['spindle_windows']
            num_spindles = np.shape(spindle_windows)[1]


            rs_indices = []
            for i in range(0, num_spindles):
                for j in range(0, num_ripples):
                    pk = ripple_peaks[j]

                    if spindle_windows[0, i] < pk < spindle_windows[1, i]:
                        rs_indices.append(j)


            ripple_windows = ripple_windows[:,rs_indices]





            pyr_spike_phases = unit_utils.spike_phases(tetrodes, ripple_phases, ripple_windows, ts, pyr_units)
            int_spike_phases = unit_utils.spike_phases(tetrodes, ripple_phases, ripple_windows, ts, int_units)


            # calculate phase locking
            for spike_phases in pyr_spike_phases:

                count, _ = np.histogram(spike_phases, bins)
                p_value, z_value = pycircstat.rayleigh(bin_centers, count, bin_widths[0])

                mean = pycircstat.mean(spike_phases)
                # wrap data from -pi to pi
                if mean > np.pi:
                    mean = mean - 2 * np.pi

                mean = mean * 180 / np.pi

                phase_info['session'].append(sessions[d])
                phase_info['strain'].append(strain)
                phase_info['p_value'].append(p_value)
                phase_info['z_value'].append(np.log(z_value))
                phase_info['mean'].append(mean)
                phase_info['class'].append('PYR')
                if p_value < 0.05 / len(pyr_spike_phases):
                    phase_info['sig'].append(True)
                else:
                    phase_info['sig'].append(False)

            for spike_phases in int_spike_phases:

                count, _ = np.histogram(spike_phases, bins)
                p_value, z_value = pycircstat.rayleigh(bin_centers, count, bin_widths[0])

                mean = pycircstat.mean(spike_phases)
                # wrap data from -pi to pi
                if mean > np.pi:
                    mean = mean - 2 * np.pi

                mean = mean * 180 / np.pi

                phase_info['session'].append(sessions[d])
                phase_info['strain'].append(strain)
                phase_info['p_value'].append(p_value)
                phase_info['z_value'].append(np.log(z_value))
                phase_info['mean'].append(mean)
                phase_info['class'].append('INT')
                if p_value < 0.05 / len(int_spike_phases):
                    phase_info['sig'].append(True)
                else:
                    phase_info['sig'].append(False)

df = pd.DataFrame(phase_info)
print(df)