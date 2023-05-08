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

from scipy import stats
import subjects

# calculate bins, 16 bins
bins = np.linspace(-np.pi, np.pi, 33)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = bins[1:] - bins[:-1]


def append_data(phase_info, units, session, task, strain, neuron_class, animal_id, phase_stats, phase_ppc, spike_phases):
    for j in range(0, len(units)):
        s = phase_stats[j]
        phase_info['session'].append(session)
        phase_info['task'].append(task)
        phase_info['strain'].append(strain)
        phase_info['p_value'].append(s[0])
        phase_info['z_value'].append(s[1])
        phase_info['mean'].append(s[2])
        phase_info['class'].append(neuron_class)
        phase_info['sig'].append(True if s[0] < 0.05 else False)
        phase_info['id'].append(animal_id)
        phase_info['ppc'].append(phase_ppc[j])
        phase_info['num_spikes'].append(len(spike_phases[j]))

    return phase_info

new = True
if os.path.exists('../output/phase_info_ripple_limited.csv') and new == False:
    print("loading")
else:
    phase_info = {
        'session': [],
        'p_value': [],
        'z_value': [],
        'task': [],
        'strain': [],
        'sig': [],
        'mean': [],
        'class': [],
        'id': [],
        'ppc': [],
        'num_spikes': []
    }

    ids = ['18-1', '25-10', '36-3', '36-1', '64-30', '59-2', '62-1']

    for id in ids:
        print('Working on ' + id)
        subject = subjects.subjects[id]
        strain = subject['strain']

        data = subjects.load_data(subject, units=True, phases=True)

        for task in subjects.sessions:

            pyr_units = data[task]['pyr_units']
            int_units = data[task]['int_units']


            for session in subjects.sessions[task]:

                ripple_phases = data[task][session]['ripple_phases']
                ripple_windows = data[task][session]['ripple_windows']
                ts = data[task][session]['ts']

                # limit ripple_windows to those that occur within first hour
                limit = 2000 * 60 * 60
                if ts.size < limit:
                    limit = ts.size

                # less that ts[2000 * 60 * 60]
                end = (ripple_windows[1,:] < limit).nonzero()
                ripple_windows = ripple_windows[:,end[0]]
                pyr_spike_phases = unit_utils.spike_phases(ripple_phases, ripple_windows,
                                                           ts,
                                                           pyr_units)


                int_spike_phases = unit_utils.spike_phases(ripple_phases, ripple_windows,
                                                           ts,
                                                           int_units)


                pyr_phase_stats = unit_utils.phase_stats(pyr_spike_phases, bins, bin_centers, bin_widths)
                pyr_phase_ppc = unit_utils.ppc(pyr_spike_phases)

                phase_info = append_data(phase_info, pyr_units, session, task, strain, 'PYR', id, pyr_phase_stats, pyr_phase_ppc, pyr_spike_phases)

                int_phase_stats = unit_utils.phase_stats(int_spike_phases, bins, bin_centers, bin_widths)
                int_phase_ppc = unit_utils.ppc(int_spike_phases)

                phase_info = append_data(phase_info, int_units, session, task, strain, 'INT', id, int_phase_stats, int_phase_ppc, int_spike_phases)



    phase_df = pd.DataFrame(phase_info)
    phase_df.to_csv('../output/phase_info_hour.csv')



