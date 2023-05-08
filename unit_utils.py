import scipy.io as sio
import pandas as pd
import os
import glob
import numpy as np
from sklearn.decomposition import FastICA, fastica
from scipy import stats
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import csv
import pycircstat
#from neo.core import SpikeTrain
#from quantities import ms
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans




def good_units(directory, rating_file, tetrodes):

    units = []

    ratings = pd.read_csv(os.path.join(directory, rating_file))


    # look at each tetrode, get dict of acceptable units (rating A or B)
    for tt in tetrodes:
        tt_ratings = ratings.iloc[:, tt - 1]

        good_clusters = tt_ratings.index[(tt_ratings == 'A') | (tt_ratings == 'B')].tolist()

        for c in good_clusters:
            summary_path = glob.glob(directory + '/TT/TT*{}c_ClusterSummary_{}.mat'.format(tt, c + 1))[0]
            unit_path = glob.glob(directory + '/TT/TT*{}c_{}__.mat'.format(tt, c + 1))[0]
            cluster_info = sio.loadmat(summary_path)

            # clean up cluster info

            units.append({
                'tetrode': tt,
                'cluster': c + 1,
                'nSpikes': cluster_info['CI'][0, 0]['nSpikes'].flatten()[0],
                'PeakToTroughPts': cluster_info['CI'][0, 0]['PeakToTroughPts'].flatten(),
                'AutoCorr': cluster_info['CI'][0, 0]['AutoCorr'].flatten(),
                'AutoCorrXAxisMsec': cluster_info['CI'][0, 0]['AutoCorrXAxisMsec'].flatten(),
                'TS': load_spikes(unit_path).flatten() * 100,
                'wf': cluster_info['CI'][0, 0]['MeanWaveform']
            })

    return units

# 1 = int
# 0 = pyr
def unit_types(directory, units):

    utype = np.zeros(len(units))

    inter = set()
    with open(directory, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            tt = row[0]
            c = row[1]
            inter.add((int(tt), int(c)))

    for i in range(0, len(units)):
        unit = units[i]
        if (unit['tetrode'], unit['cluster']) in inter:
            utype[i] = 1


    return utype

def split_unit_types(directory, units):

    # interneurons
    inter = set()
    with open(directory, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            tt = row[0]
            c = row[1]
            inter.add((int(tt),int(c)))

    pyr_units = []
    int_units = []

    for unit in units:
        if (unit['tetrode'], unit['cluster']) in inter:
            int_units.append(unit)
        else:
            pyr_units.append(unit)

    return pyr_units, int_units

def load_spikes(filename, start=None, end=None):

    data = sio.loadmat(filename)['TS']

    # if start and end given, crop timestamp

    if start is not None and end is not None:
        data = data[(data >= start) & (data <= end)]

    return data


def spikes_in_window(data, start, end):
    return data[(data >= start) & (data <= end)]

def spikes_in_windows(data, windows):
    all_data = []
    for w in windows.T:
        all_data.append(data[(data >= w[0]) & (data <= w[1])])

    return all_data

def pca(z):
    cov_matrix = np.cov(z)
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)

    sorted_ind = np.argsort(eig_values)[::-1]
    eig_vectors = eig_vectors[:, sorted_ind]
    eig_values = eig_values[sorted_ind]

    return eig_values, eig_vectors

def assembly_templates_ica(z_matrix):

    num_bins = z_matrix.shape[1]

    eig_values, eig_vectors = pca(z_matrix)
    #plt.stem(eig_values)
    #plt.show()


    # marcenkopastur
    q = num_bins / eig_values.size
    thresh = ((1 + np.sqrt(1 / q)) ** 2)

    num_assemblies = (eig_values > thresh).sum()
    if num_assemblies == 0:
        return []

    """    # project onto significant PCs
    sig_pcs = eig_vectors[eig_values > thresh]
    projection = sig_pcs @ z_matrix
    # print(projection.shape)

    # calculate ICA using sklearn fastICA
    transformer = FastICA(n_components=num_assemblies, whiten='unit-variance', max_iter=500)
    templates = transformer.fit_transform(z_matrix)"""

    # fastica
    _, _, templates = fastica(z_matrix, n_components=num_assemblies, whiten='arbitrary-variance', max_iter=500, tol=1e-04, compute_sources=True)

    return templates


# find assemblies by projection strength
def assembly_detection(templates, z_matrix):
    # projection matrix
    num_assemblies = templates.shape[1]
    num_bins = z_matrix.shape[1]

    time_proj = np.zeros((num_assemblies, num_bins - 1))

    for i in range(0, num_assemblies):

        P = np.outer(templates[:, i], templates[:, i])
        np.fill_diagonal(P, 0)

        for t in range(0, num_bins - 1):
            strength = z_matrix[:, t] @ P @ z_matrix[:, t]
            time_proj[i, t] = strength

    return time_proj


def spike_trains(directory, units):
    ripple_data = sio.loadmat(os.path.join(directory, "ripples.mat"))
    ts = ripple_data['ts'].flatten()

    start = ts[0]
    stop = ts[-1]

    spikes = []

    for i in range(0, len(units)):
        spike_ts = units[i]['TS']

        # bin spikes
        s = spikes_in_window(spike_ts, start, stop)
        s = s/1000
        spikes.append(s)

    # return spike times in milliseconds
    return spikes

"""def ripple_spike_trains(directory, units):
    ripple_data = sio.loadmat(os.path.join(directory, "ripples.mat"))
    ripple_windows = merge_ripples(ripple_data['ripple_windows'])


    ts = ripple_data['ts'].flatten()
    num_ripples = np.shape(ripple_windows)[1]

    trains = []

    for j in range(0, num_ripples):
        spikes = []
        start = ts[ripple_windows[0,j]]
        stop = ts[ripple_windows[1,j]]
        for i in range(0, len(units)):

            spike_ts = units[i]['TS']

            # bin spikes
            s = spikes_in_window(spike_ts, start, stop)
            s = s / 1000
            st = SpikeTrain(s * ms, t_start=start/1000, t_stop=stop/1000)
            spikes.append(st)
        trains.append(spikes)

    # return spike times in milliseconds
    return trains"""


# use ripple windows to define start and stop timestamps to add spikes
# if binsize is none, use full ripple
def ripple_spike_matrix(ts, ripple_windows, units, binsize=None):
    spike_windows = []
    for i in range(ripple_windows.shape[1]):
        spike_windows.append((ts[ripple_windows[0, i]], ts[ripple_windows[1, i]]))

    spike_windows = np.array(spike_windows).T

    spike_matrix = []

    for unit in units:
        spike_ts = unit['TS']

        spikes = spikes_in_windows(spike_ts, spike_windows)
        # need to normalize by duration of ripple
        num_spikes = [len(s) for s in spikes]
        spike_matrix.append(num_spikes)

    spike_matrix = np.array(spike_matrix)
    z = stats.zscore(spike_matrix, axis=1)

    z = np.nan_to_num(z)
    return z



# binned spike matrix
# binsize in ms
def spike_matrix(ts, units, binsize):

    start = ts[0]
    stop = ts[-1]

    # binsize in ms
    binsize = binsize * 1000

    # bins until end of timepoint -> exclude last
    bins = np.arange(start, stop + binsize, binsize)

    spike_matrix = []

    # define time bins: 1 ms bins = 1000 us
    # ts in us

    for unit in units:
        spike_ts = unit['TS']

        s = spikes_in_window(spike_ts, start, bins[-1])
        binned, edges = np.histogram(s, bins)
        spike_matrix.append(binned)


    spike_matrix = np.array(spike_matrix)

    return spike_matrix, bins

# binsize in ms
# TODO: deal with nan somehow
def z_spike_matrix(ts, units, binsize):

    s, bins = spike_matrix(ts, units, binsize)
    z = stats.zscore(s, axis=1)

    z = np.nan_to_num(z)
    return z, bins

# return bin indices for ripples
# for each ripple, find start and stop bin and add to list
def ripple_bins(bins, ripple_windows, ts):
    num_ripples = np.shape(ripple_windows)[1]
    r_start_bins = []
    r_bins = []

    for i in range(0, num_ripples):
        # find ind
        r_start = ts[ripple_windows[0, i]]
        r_end = ts[ripple_windows[1, i]]

        # find bins
        start_bin = np.searchsorted(bins, r_start, side='right')
        end_bin = np.searchsorted(bins, r_end, side='right')

        r_start_bins.append(start_bin)
        for j in range(start_bin,end_bin+1):
            r_bins.append(j)

    return np.unique(r_bins), r_start_bins



def merge_ripples(ripple_windows):
    num_ripples = np.shape(ripple_windows)[1]
    j = 0
    rs = ripple_windows[0, j]
    re = ripple_windows[1, j]
    ripples = []
    while j < num_ripples - 1:

        # 40 = 20 ms
        if ripple_windows[0, j + 1] - re > 40:
            ripples.append((rs, re))
            rs = ripple_windows[0, j + 1]
            re = ripple_windows[1, j + 1]
        else:
            re = ripple_windows[1, j + 1]
        j += 1

    ripple_windows = np.array(ripples).T

    return ripple_windows

def spike_phases(ripple_phases, ripple_windows,ts, units):
    num_ripples = np.shape(ripple_windows)[1]
    spike_phases = []
    for unit in units:

        # get specific tetrode ripple
        tt = unit['tetrode']
        #tt_array = np.array(tetrodes)
        #lfp_num = np.where(tt_array == unit['tetrode'])


        ripple_phase = ripple_phases['CSC0' + str(tt)]
        # Check shape
        """if (ripple_phases.shape[0] > 1):
            ripple_phase = ripple_phases[lfp_num][0].flatten()
        else:
            ripple_phase = ripple_phases"""

        spike_ts = unit['TS']
        phases = []

        for j in range(0, num_ripples):

            w_start = ripple_windows[0, j]
            w_end = ripple_windows[1, j]
            spikes = spikes_in_window(spike_ts, ts[w_start], ts[w_end])

            ripple_ts = ts[w_start:w_end]

            for s in spikes:
                # find phase of spike
                phases.append(ripple_phase[np.absolute(ripple_ts - s).argmin() + w_start])
        spike_phases.append(np.array(phases))

    return spike_phases

def spike_phases_spindles(spindle_phases, spindle_windows,ts, units):
    num_spindles = np.shape(spindle_windows)[1]
    spike_phases = []
    for unit in units:

        # get specific tetrode ripple
        spindle_phase = spindle_phases

        spike_ts = unit['TS']
        phases = []

        for j in range(0, num_spindles):

            w_start = spindle_windows[0, j]
            w_end = spindle_windows[1, j]
            spikes = spikes_in_window(spike_ts, ts[w_start], ts[w_end])

            ripple_ts = ts[w_start:w_end]

            for s in spikes:
                # find phase of spike
                phases.append(spindle_phase[np.absolute(ripple_ts - s).argmin() + w_start])
        spike_phases.append(np.array(phases))

    return spike_phases

def phase_stats(spike_phases, bins, bin_centers, bin_widths):
    stats = []
    for j in range(0, len(spike_phases)):
        s = spike_phases[j]
        count, _ = np.histogram(s, bins)

        # rayleigh test for uniformity
        # plt.figure(i)
        # ax = plt.subplot(polar=True)
        # bars = ax.bar(bin_centers, count, width=bin_widths, bottom=0.0)
        # plt.show()

        p_value, z_value = pycircstat.rayleigh(bin_centers, count, bin_widths[0])

        # calculate median
        mean = pycircstat.mean(s)
        # wrap data from -pi to pi
        if mean > np.pi:
            mean = mean - 2 * np.pi

        # convert to deg
        mean = mean * 180 / np.pi
        stats.append((p_value, z_value, mean))

    return stats

def ppc(phases):
    # avg of cosine of all the differences

    ppcs = []

    for k in range(0, len(phases)):
        s = phases[k]
        sum = 0
        counter = 0
        for i in range(0, len(s)):
            for j in range(i,len(s)):
                sum += np.cos(s[j]) * np.cos(s[i]) + np.sin(s[j]) * np.sin(s[i])
                counter += 1
        ppcs.append(np.float64(sum)/np.float64(counter))

    return ppcs

def reactivation_indices(strengths):

    ind = []

    for i in range(strengths.shape[0]):
        s = strengths[i,:]
        ind.append(np.argwhere(s>5).flatten())

    return ind

def extract_cell_numbers(templates):
    # for each template
    # return cells using kmeans clustering

    cells = []
    #X = templates.flatten()
    # cluster all data points


    for i in range(0, templates.shape[1]):



        X = templates[:, i]

        t1 = np.mean(X) + 2 * np.std(X)
        t2 = np.mean(X) - 2 * np.std(X)
        #threshold = 0.2

        # try 0.2 as threshold and take largest group, if tie, take group with largest index
        larger = np.argwhere(X > t1).flatten()
        smaller = np.argwhere(X < t2).flatten()

        indices = []

        if larger.size > smaller.size:
            indices = larger
        elif smaller.size > larger.size:
            indices = smaller
        else:
            # get max
            if np.argmax(X) in larger:
                indices = larger
            else:
                indices = smaller


        labels = np.zeros(X.size)
        labels[indices] = 1.0
        cells.append(labels)

        #try kde
        #stats.kde(X)
        #kernel = stats.gaussian_kde(X)

        #np.linspace(0,1,10)

        """# get max index
        max_ind = np.argmax(np.abs(X))
        kmeans = KMeans(n_clusters=2).fit(X.reshape(-1, 1))
        labels = kmeans.labels_
        #labels = kmeans.predict(X.reshape(-1, 1))

        # if label has 0 for max ind, flip
        if labels[max_ind] == 0:
            labels = 1-labels"""

        #cells.append(labels)

    cells = np.array(cells)


    return cells.T

def activation_correlation(num_templates, z_strengths, activations, ripple_bins):

    max_corr = []
    diff = []
    corr = []
    lags = []

    for i in range(0, num_templates):
        activation_signal = np.zeros(z_strengths.shape[1])
        activation_signal[activations[i]] = 1
        ripple_signal = np.zeros(z_strengths.shape[1])
        ripple_signal[ripple_bins] = 1

        activation_signal = activation_signal / np.linalg.norm(activation_signal)
        ripple_signal = ripple_signal / np.linalg.norm(ripple_signal)

        # try correlating with strengths
        z_signal = z_strengths[i,:] / np.linalg.norm(z_strengths[0,:])

        c = signal.correlate(activation_signal, ripple_signal)
        lag = signal.correlation_lags(len(activation_signal), len(ripple_signal))

        corr.append(c)
        lags.append(lag)

        #min = int(lags.size / 2) - 30
        #max = int(lags.size / 2) + 30
        # plt.plot(lags[min:max], corr[min:max])


        max_c = -1
        max_d = -1
        for v in c:
            if v > max_c:
                max_d = max_c
                max_c = v



        max_corr.append(max_c)
        diff.append(max_c-max_d)


        # remove c


    return corr, lags, max_corr, diff


# given unit firing times, bin into 1 ms seconds
# apply gaussian 20 ms std, 60ms halfwidth
# note: only apply this to pyramidal neurons
def identifyUnitBursts(directory,units, binsize=1, sigma=20,halfwidth=60):

    # calculate spike matrix
    s = spike_matrix(directory, units, binsize)

    # compress unit count

    # apply gaussian
    filtered = gaussian_filter1d(s,sigma=sigma, axis=1)

