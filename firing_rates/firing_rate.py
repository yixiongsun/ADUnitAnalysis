import unit_utils
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import subjects
import pandas as pd
import os
import pickle

# return swr activated and inhibited neurons
# 10 ms binsize
def swr_modulated(units, sample_rate, ts=[], ripple_windows=[], binsize=10, spike_windows=[]):
    binsize = binsize * 1000
    unit_id = []
    strengths = []
    modulations = []

    all_spikes = []

    # loop through ripple windows and find ripple onset



    if len(spike_windows) == 0:
        # balance, if sleep 2 or 3 only look at first hour to control for recording duration
        ripple_onsets = ripple_windows[0, :]


        for i, r in enumerate(ripple_onsets):
            # 500 ms before and 500 ms after
            # sample rate 2000
            start = r - int(sample_rate * 0.5)
            stop = r + int(sample_rate * 0.5)

            # if start is > 1 hour = sample_rate * 60 * 60
            if start > sample_rate * 60 * 60:
                break

            if start >= 0 and stop < len(ts):
                # check for ts
                #something wrong with data
                if ts[stop] - ts[start] != 1000000:
                    spike_windows.append((ts[r] - 500000, ts[r] + 500000))
                else:
                    spike_windows.append((ts[start], ts[stop]))

        spike_windows = np.array(spike_windows).T



    for unit in units:
        spike_ts = unit['TS']

        spikes = unit_utils.spikes_in_windows(spike_ts, spike_windows)

        num_spikes = sum([len(s) for s in spikes])
        if num_spikes < 50:
            unit_id.append(str(unit['tetrode'])  + '_' + str(unit['cluster']))
            strengths.append(0)
            modulations.append('N/A')
            continue

        # binsize in ms


        # bin s
        spike_matrix = []
        for i in range(spike_windows.shape[1]):
            bins = np.arange(spike_windows[0, i], spike_windows[1, i] + binsize, binsize)

            binned, edges = np.histogram(spikes[i], bins)
            spike_matrix.append(binned)
        spike_matrix = np.array(spike_matrix)

        #sns.heatmap(spike_matrix)
        #plt.show()

        #filtered_spike_matrix = gaussian_filter1d(spike_matrix, sigma=sigma, axis=1)
        mean_fr = np.mean(spike_matrix, axis=0)

        # zscore based on first 200 ms (first 20 bins
        zscore_fr = stats.zmap(mean_fr, mean_fr[0:20])

        #plt.plot(zscore_fr)
        #plt.show()
        # zscore_fr = stats.zscore(mean_fr)

        r_bin = int(len(zscore_fr) / 2)
        # fr modulation strength = avg zscore in 50 ms window (5 bins)
        strength = np.mean(zscore_fr[r_bin:r_bin + 5])


        # threshold 2 SD = modulated 200 ms (20 bins)
        increased = np.argwhere(zscore_fr[r_bin:r_bin+20] > 2)
        modulation = 'none'
        mod_ind = np.inf
        if len(increased) > 0:
            modulation = 'increased'
            mod_ind = increased[0,0]

        decreased = np.argwhere(zscore_fr[r_bin:r_bin + 20] < -2)
        if len(decreased) > 0:
            if decreased[0,0] < mod_ind:
                modulation = 'decreased'

        unit_id.append(str(unit['tetrode'])  + '_' + str(unit['cluster']))
        strengths.append(strength)
        modulations.append(modulation)

        all_spikes.append(zscore_fr)

    all_spikes = np.array(all_spikes)

    max_binned = np.max(all_spikes[:, 50:55], 1)

    sorted_ind = np.argsort(max_binned)[::-1]
    #sns.heatmap(all_spikes[sorted_ind])
    #plt.show()

    return unit_id, strengths, modulations

# combined modulation for sleep 1 and sleep 2
def swr_modulation_combined(units, sleep_data, sample_rate):
    # call prev function but use spike windows of sleep 1 and 2

    spike_windows = []
    for s in ['Sleep1', 'Sleep2']:
        ripple_windows = sleep_data[s]['ripple_windows']
        ts = sleep_data[s]['ts']

        ripple_onsets = ripple_windows[0, :]

        for i, r in enumerate(ripple_onsets):
            # 500 ms before and 500 ms after
            # sample rate 2000
            start = r - int(sample_rate * 0.5)
            stop = r + int(sample_rate * 0.5)
            if start >= 0 and stop < len(ts):

                if ts[stop] - ts[start] != 1000000:
                    spike_windows.append((ts[r] - 500000, ts[r] + 500000))
                else:
                    spike_windows.append((ts[start], ts[stop]))

    spike_windows = np.array(spike_windows).T

    return swr_modulated(units, sample_rate, spike_windows=spike_windows)

# similar to above but return only details for raster plot, save in pickle file
def swr_raster(units, sleep_data, sample_rate, binsize=10):
    all_spikes = []
    spike_windows = []
    for s in ['Sleep1', 'Sleep2']:
        ripple_windows = sleep_data[s]['ripple_windows']
        ts = sleep_data[s]['ts']

        ripple_onsets = ripple_windows[0, :]

        for i, r in enumerate(ripple_onsets):
            # 500 ms before and 500 ms after
            # sample rate 2000
            start = r - int(sample_rate * 0.5)
            stop = r + int(sample_rate * 0.5)
            if start >= 0 and stop < len(ts):
                if ts[stop] - ts[start] != 1000000:
                    spike_windows.append((ts[r] - 500000, ts[r] + 500000))
                else:
                    spike_windows.append((ts[start], ts[stop]))

    spike_windows = np.array(spike_windows).T

    binsize = binsize * 1000

    for unit in units:
        spike_ts = unit['TS']

        spikes = unit_utils.spikes_in_windows(spike_ts, spike_windows)


        # binsize in ms


        # bin s
        spike_matrix = []
        for i in range(spike_windows.shape[1]):
            bins = np.arange(spike_windows[0, i], spike_windows[1, i] + binsize, binsize)

            binned, edges = np.histogram(spikes[i], bins)
            spike_matrix.append(binned)
        spike_matrix = np.array(spike_matrix)
        mean_fr = np.mean(spike_matrix, axis=0)

        # divide by bin size after number of ripples
        # 10 ms bin = 0.01 s

        mean_fr = mean_fr / (binsize / 1000000)

        all_spikes.append(mean_fr)

    all_spikes = np.array(all_spikes)
    return all_spikes

# calculate firing rates, use full timestamp of sleep 1 and sleep 2 for hab and olm
def firing_rate(units, ripple_windows, ts):


    total_duration = (ts[-1] - ts[0]) / 1000000

    unit_id = []
    total_r_spikes = []
    total_r_duration = []
    total_n_r_spikes = []
    total_n_r_duration = []


    for unit in units:
        spike_ts = unit['TS']
        total_spikes = unit_utils.spikes_in_window(spike_ts, ts[0], ts[-1]).size

        r_spikes = 0
        r_duration = 0

        for j in range(0, ripple_windows.shape[1]):
            start = ts[ripple_windows[0, j]]
            end = ts[ripple_windows[1, j]]

            ripple_spikes = unit_utils.spikes_in_window(spike_ts, start, end).size

            r_spikes += ripple_spikes
            r_duration += (end - start) / 1000000

        n_r_spikes = total_spikes - r_spikes
        n_r_duration = total_duration - r_duration

        unit_id.append(str(unit['tetrode']) + '_' + str(unit['cluster']))
        total_r_spikes.append(r_spikes)
        total_r_duration.append(r_duration)
        total_n_r_spikes.append(n_r_spikes)
        total_n_r_duration.append(n_r_duration)

    return unit_id, total_r_spikes, total_r_duration, total_n_r_spikes, total_n_r_duration


if __name__ == "__main__":

    # run pipeline
    #ids = ['18-1', '25-10', '36-3', '36-1', '64-30','59-2', ]
    ids = ['62-1']
    for id in ids:
        subject = subjects.subjects[id]

        data = subjects.load_data(subject, units=True)

        for task in subjects.sessions:
            unit_id, strengths, modulations = swr_modulation_combined(data[task]['pyr_units'], data[task], data[task]['sample_rate'])
            df_pyr = pd.DataFrame(
                {'unit': unit_id, 'strengths': strengths, 'modulation': modulations, 'class': ['pyr'] * len(unit_id),
                 'id': [id] * len(unit_id), 'strain': [subject['strain']] * len(unit_id), 'task': [task] * len(unit_id)})


            unit_id, strengths, modulations = swr_modulation_combined(data[task]['int_units'], data[task], data[task]['sample_rate'])
            df_int = pd.DataFrame(
                {'unit': unit_id, 'strengths': strengths, 'modulation': modulations, 'class': ['int'] * len(unit_id),
                 'id': [id] * len(unit_id), 'strain': [subject['strain']] * len(unit_id), 'task': [task] * len(unit_id)})

            df = pd.concat([df_pyr, df_int])

            df.to_csv(os.path.join(subject[task + 'dir'], 'swr_modulation.csv'))

            pyr_spikes = swr_raster(data[task]['pyr_units'], data[task], data[task]['sample_rate'], binsize=10)
            int_spikes = swr_raster(data[task]['int_units'], data[task], data[task]['sample_rate'], binsize=10)
            pickle.dump((pyr_spikes, int_spikes), open(os.path.join(subject[task + 'dir'], 'binned_fr.pkl'), 'wb'))



            for session in subjects.sessions[task]:
                # task = hab/olm, session = sleep
                print(id + task + session)


                unit_id, strengths, modulations = swr_modulated(data[task]['pyr_units'], data[task]['sample_rate'], ripple_windows=data[task][session]['ripple_windows'], ts=data[task][session]['ts'], spike_windows=[])

                df_pyr = pd.DataFrame(
                    {'unit': unit_id, 'strengths': strengths, 'modulation': modulations, 'class': ['pyr'] * len(unit_id), 'id': [id] * len(unit_id), 'strain': [subject['strain']] * len(unit_id), 'task': [task] * len(unit_id), 'session': [session] * len(unit_id)})
                unit_id, strengths, modulations = swr_modulated(data[task]['int_units'], data[task]['sample_rate'], ripple_windows=data[task][session]['ripple_windows'], ts=data[task][session]['ts'], spike_windows=[])
                df_int = pd.DataFrame(
                    {'unit': unit_id, 'strengths': strengths, 'modulation': modulations, 'class': ['int'] * len(unit_id), 'id': [id] * len(unit_id), 'strain': [subject['strain']] * len(unit_id), 'task': [task] * len(unit_id), 'session': [session] * len(unit_id)})

                df = pd.concat([df_pyr, df_int])

                df.to_csv(os.path.join(subject[task + 'dir'], session, 'swr_modulation.csv'))



                # firing rates
                unit_id, total_r_spikes, total_r_duration, total_n_r_spikes, total_n_r_duration = firing_rate(data[task]['pyr_units'], data[task][session]['ripple_windows'], data[task][session]['ts'])
                df_pyr = pd.DataFrame(
                    {'unit': unit_id + unit_id, 'spikes': total_r_spikes + total_n_r_spikes, 'duration': total_r_duration + total_n_r_duration,
                     'class': ['pyr'] * 2 * len(unit_id),
                     'id': [id] * 2 * len(unit_id), 'strain': [subject['strain']] * 2 * len(unit_id), 'task': [task] * 2 * len(unit_id),
                     'session': [session] * 2 * len(unit_id), 'Interval': ['Ripple'] * len(unit_id) + ['Non Ripple'] * len(unit_id)})

                unit_id, total_r_spikes, total_r_duration, total_n_r_spikes, total_n_r_duration = firing_rate(
                    data[task]['int_units'], data[task][session]['ripple_windows'], data[task][session]['ts'])
                df_int = pd.DataFrame(
                    {'unit': unit_id + unit_id, 'spikes': total_r_spikes + total_n_r_spikes, 'duration': total_r_duration + total_n_r_duration,
                     'class': ['int'] * 2 * len(unit_id),
                     'id': [id] * 2 * len(unit_id), 'strain': [subject['strain']] * 2 * len(unit_id), 'task': [task] * 2 * len(unit_id),
                     'session': [session] * 2 * len(unit_id), 'Interval': ['Ripple'] * len(unit_id) + ['Non Ripple'] * len(unit_id)})

                df = pd.concat([df_pyr, df_int])

                df.to_csv(os.path.join(subject[task + 'dir'], session, 'firing_rates.csv'))