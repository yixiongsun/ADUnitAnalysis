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

from scipy import stats

# calculate bins, 16 bins
bins = np.linspace(-np.pi, np.pi, 33)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = bins[1:] - bins[:-1]

sessions = ["Hab Rec 1", "Hab Rec 2", "OLM Rec 1", "OLM Rec 2", "OLM Rec 3", "Hab Rec 1", "Hab Rec 2", "OLM Rec 1",
                "OLM Rec 2", "OLM Rec 3"]
strains = ["TG", "TG", "TG", "TG", "TG","WT", "WT", "WT", "WT", "WT"]

new = True
if os.path.exists('../output/spindle_phase_modulation.pkl') and new == False:
    print("loading")
    (pyr_phases, int_phases) = pickle.load(open('../output/spindle_phase_modulation.pkl', 'rb'))
else:
    directories = [
        'D:\\TetrodeData\\2022-07-08_10-01-42-p\\Sleep1',
        'D:\\TetrodeData\\2022-07-08_10-01-42-p\\Sleep2',
        'D:\\TetrodeData\\2022-07-09_09-54-32-p\\Sleep1',
        'D:\\TetrodeData\\2022-07-09_09-54-32-p\\Sleep2',
        'D:\\TetrodeData\\2022-07-09_09-54-32-p\\Sleep3',
        'D:\\TetrodeData\\2022-07-10_13-27-33-p\\Sleep1',
        'D:\\TetrodeData\\2022-07-10_13-27-33-p\\Sleep2',
        'D:\\TetrodeData\\2022-07-11_13-47-33-p\\Sleep1',
        'D:\\TetrodeData\\2022-07-11_13-47-33-p\\Sleep2',
        'D:\\TetrodeData\\2022-07-11_13-47-33-p\\Sleep3'
    ]

    rating_files = [
        'D:\\TetrodeData\\2022-07-08_10-01-42-p\\ClusterRating_2022-07-08_10-01-42-p.csv',
        'D:\\TetrodeData\\2022-07-08_10-01-42-p\\ClusterRating_2022-07-08_10-01-42-p.csv',
        'D:\\TetrodeData\\2022-07-09_09-54-32-p\\ClusterRating_2022-07-09_09-54-32-p.csv',
        'D:\\TetrodeData\\2022-07-09_09-54-32-p\\ClusterRating_2022-07-09_09-54-32-p.csv',
        'D:\\TetrodeData\\2022-07-09_09-54-32-p\\ClusterRating_2022-07-09_09-54-32-p.csv',
        'D:\\TetrodeData\\2022-07-10_13-27-33-p\\ClusterRating_2022-07-10_13-27-33-p.csv',
        'D:\\TetrodeData\\2022-07-10_13-27-33-p\\ClusterRating_2022-07-10_13-27-33-p.csv',
        'D:\\TetrodeData\\2022-07-11_13-47-33-p\\ClusterRating_2022-07-11_13-47-33-p.csv',
        'D:\\TetrodeData\\2022-07-11_13-47-33-p\\ClusterRating_2022-07-11_13-47-33-p.csv',
        'D:\\TetrodeData\\2022-07-11_13-47-33-p\\ClusterRating_2022-07-11_13-47-33-p.csv',
    ]




    tetrodes = [[1, 2, 3, 5, 7, 8], [1, 2, 3, 5, 7, 8], [1, 2, 3, 5, 7, 8], [1, 2, 3, 5, 7, 8], [1, 2, 3, 5, 7, 8],
                [2, 3, 5, 6, 7, 8], [2, 3, 5, 6, 7, 8], [2, 3, 5, 6, 7, 8], [2, 3, 5, 6, 7, 8], [2, 3, 5, 6, 7, 8]]

    pyr_phases = []
    int_phases = []
    non_concat_phases = []

    for k in range(0, len(directories)):

        rating_file = rating_files[k]
        directory = directories[k]

        units = unit_utils.good_units(os.path.dirname(directory), rating_file, tetrodes[k])

        # load ripple timestamps
        spindle_data = sio.loadmat(os.path.join(directory, "spindles.mat"))

        spindle_phases = spindle_data['spindle_phases'].flatten()

        spindle_windows = spindle_data['spindle_windows']
        ts = spindle_data['ts'].flatten()
        num_spindles = np.shape(spindle_windows)[1]

        # split by unit type
        pyr_units, int_units = unit_utils.split_unit_types(os.path.join(os.path.dirname(rating_file), 'putative_interneurons.csv'), units)

        pyr_spike_phases = unit_utils.spike_phases_spindles(spindle_phases, spindle_windows, ts, pyr_units)
        int_spike_phases = unit_utils.spike_phases_spindles(spindle_phases, spindle_windows, ts, int_units)

        pyr_phases.append(pyr_spike_phases)
        int_phases.append(int_spike_phases)


    pickle.dump((pyr_phases, int_phases), open('../output/spindle_phase_modulation.pkl', 'wb'))


# rayleigh test: unimodal
# omnibus test: can be multiple modes
sns.set_style('ticks')
new = True
if os.path.exists('../output/spindle_phase_info.csv') and os.path.exists('../output/spindle_sig_phase_info.csv') and new == False:
    print("loading")
    phase_df = pd.read_csv('../output/spindle_phase_info.csv')
    sig_units_df = pd.read_csv('../output/spindle_sig_phase_info.csv')

else:
    phase_info = {
        'session': [],
        'p_value': [],
        'z_value': [],
        'strain': [],
        'sig': [],
        'mean': [],
        'class': []
    }

    sig_units = {
        'session': [],
        'count': [],
        'strain': [],
        'class': []
    }

    for i in range(0, len(pyr_phases)):
        spike_phases = pyr_phases[i]



        # create subplot
        plt.figure(figsize=(15, 15))

        num_units = len(spike_phases)
        sig_count = 0



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
            mean = mean * 180/np.pi


            phase_info['session'].append(sessions[i])
            phase_info['strain'].append(strains[i])
            phase_info['p_value'].append(p_value)
            phase_info['z_value'].append(np.log(z_value))
            phase_info['mean'].append(mean)
            phase_info['class'].append('PYR')
            if p_value < 0.05/num_units:
                sig_count += 1
                phase_info['sig'].append(True)
                #ax = plt.subplot(6, 6, sig_count, polar=True)
                #bars = ax.bar(bin_centers, count, width=bin_widths, bottom=0.0, linewidth=0, edgecolor='black', alpha=0.8)
                #plt.xticks(fontsize=7)
                #plt.yticks(fontsize=7)
            else:
                phase_info['sig'].append(False)

        #plt.suptitle("PYR Phase locked units ({}, {})".format(strains[i], sessions[i]))
        #plt.tight_layout()
        #plt.savefig("output/spindle_phase_modulation_analysis/pyr_" + sessions[i] + strains[i] + "sig_bonferroni.png", dpi=300)
        sig_units['session'].append(sessions[i])
        sig_units['strain'].append(strains[i])
        sig_units['count'].append(sig_count/num_units * 100)
        sig_units['class'].append('PYR')


    for i in range(0, len(int_phases)):
        spike_phases = int_phases[i]



        # create subplot
        plt.figure(figsize=(15, 15))

        num_units = len(spike_phases)
        sig_count = 0



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
            mean = mean * 180/np.pi


            phase_info['session'].append(sessions[i])
            phase_info['strain'].append(strains[i])
            phase_info['p_value'].append(p_value)
            phase_info['z_value'].append(np.log(z_value))
            phase_info['mean'].append(mean)
            phase_info['class'].append('INT')
            if p_value < 0.05/num_units:
                sig_count += 1
                phase_info['sig'].append(True)
                #ax = plt.subplot(6, 6, sig_count, polar=True)
                #bars = ax.bar(bin_centers, count, width=bin_widths, bottom=0.0, linewidth=0, edgecolor='black', alpha=0.8)
                #plt.xticks(fontsize=7)
                #plt.yticks(fontsize=7)
            else:
                phase_info['sig'].append(False)

        #plt.suptitle("INT Phase locked units ({}, {})".format(strains[i], sessions[i]))
        #plt.tight_layout()
        #plt.savefig("output/spindle_phase_modulation_analysis/int_" + sessions[i] + strains[i] + "sig_bonferroni.png", dpi=300)
        sig_units['session'].append(sessions[i])
        sig_units['strain'].append(strains[i])
        sig_units['count'].append(sig_count/num_units * 100)
        sig_units['class'].append('INT')


    phase_df = pd.DataFrame(phase_info)
    sig_units_df = pd.DataFrame(sig_units)
    phase_df.to_csv('../output/spindle_phase_info.csv')
    sig_units_df.to_csv('../output/spindle_sig_phase_info.csv')



"""
1. Rose plot 
TODO: Gain plot
"""

#plt.figure(figsize=(5, 8))

wt_phases_pyr = np.concatenate((np.concatenate(pyr_phases[0]), np.concatenate(pyr_phases[1]), np.concatenate(pyr_phases[2]), np.concatenate(pyr_phases[3]), np.concatenate(pyr_phases[4])))
wt_phases_int = np.concatenate((np.concatenate(int_phases[0]), np.concatenate(int_phases[1]), np.concatenate(int_phases[2]), np.concatenate(int_phases[3]), np.concatenate(int_phases[4])))
wt_phases = np.concatenate((wt_phases_pyr, wt_phases_int))

tg_phases_pyr = np.concatenate((np.concatenate(pyr_phases[5]), np.concatenate(pyr_phases[6]), np.concatenate(pyr_phases[7]), np.concatenate(pyr_phases[8]), np.concatenate(pyr_phases[9])))
tg_phases_int = np.concatenate((np.concatenate(int_phases[5]), np.concatenate(int_phases[6]), np.concatenate(int_phases[7]), np.concatenate(int_phases[8]), np.concatenate(int_phases[9])))
tg_phases = np.concatenate((tg_phases_pyr, tg_phases_int))



count_wt, _ = np.histogram(wt_phases, bins)
count_tg, _ = np.histogram(tg_phases, bins)

p_value, z_value = pycircstat.rayleigh(bin_centers, count_wt, bin_widths[0])
print(p_value, z_value)


ax = plt.subplot(2,1,1, polar=True)

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
ax.set_title('WT', fontsize=12)
bars = ax.bar(bin_centers, count_wt, width=bin_widths, bottom=0.0, linewidth=0, edgecolor='black',alpha=0.8)
#ax.text(2,2, "Z = {:.1e}, P = {:.1e}".format(z_value, p_value), fontsize=7)


ax = plt.subplot(2,1,2, polar=True)

p_value, z_value = pycircstat.rayleigh(bin_centers, count_tg, bin_widths[0])
print(p_value, z_value)

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
ax.set_title('TG', fontsize=12)
bars = ax.bar(bin_centers, count_tg, width=bin_widths, bottom=0.0, linewidth=0, edgecolor='black',alpha=0.8)
#ax.text(2,2, "Z = {:.1e}, P = {:.1e}".format(z_value, p_value), fontsize=7)
plt.subplots_adjust(hspace=0)
plt.tight_layout()
plt.savefig('../output/spindle_phase_modulation_analysis/spike_phase_all.png', dpi=300)


#int_phases[0][1]
"""count_wt, _ = np.histogram(int_phases[0][1], bins)
count_tg, _ = np.histogram(tg_phases, bins)

p_value, z_value = pycircstat.rayleigh(bin_centers, count_wt, bin_widths[0])
print(p_value, z_value)
plt.rcParams['font.size'] = '16'


ax = plt.subplot(1,1,1, polar=True)

#plt.xticks(fontsize=7)
#plt.yticks(fontsize=7)
bars = ax.bar(bin_centers, count_wt, width=bin_widths, bottom=0.0, linewidth=0, edgecolor='black',alpha=0.8)
#ax.text(2,2, "Z = {:.1e}, P = {:.1e}".format(z_value, p_value), fontsize=7)
ax.set_yticks([])
ax.set_xticks([0, np.pi/2, np.pi, 3 * np.pi/2])
plt.tight_layout()
plt.savefig('output/phase_modulation_analysis/spike_phase_example.png', dpi=300)"""





"""count_wt, _ = np.histogram(wt_phases_pyr, bins)
count_tg, _ = np.histogram(tg_phases_pyr, bins)

p_value, z_value = pycircstat.rayleigh(bin_centers, count_wt, bin_widths[0])
print(p_value, z_value)


ax = plt.subplot(2,1,1, polar=True)

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
ax.set_title('WT', fontsize=12)
bars = ax.bar(bin_centers, count_wt, width=bin_widths, bottom=0.0, linewidth=0, edgecolor='black',alpha=0.8)
#ax.text(2,2, "Z = {:.1e}, P = {:.1e}".format(z_value, p_value), fontsize=7)


ax = plt.subplot(2,1,2, polar=True)

p_value, z_value = pycircstat.rayleigh(bin_centers, count_tg, bin_widths[0])
print(p_value, z_value)

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
ax.set_title('TG', fontsize=12)
bars = ax.bar(bin_centers, count_tg, width=bin_widths, bottom=0.0, linewidth=0, edgecolor='black',alpha=0.8)
#ax.text(2,2, "Z = {:.1e}, P = {:.1e}".format(z_value, p_value), fontsize=7)
plt.subplots_adjust(hspace=0)
plt.tight_layout()
plt.savefig('output/phase_modulation_analysis/spike_phase_pyr.png', dpi=300)"""


"""count_wt, _ = np.histogram(wt_phases_int, bins)
count_tg, _ = np.histogram(tg_phases_int, bins)

p_value, z_value = pycircstat.rayleigh(bin_centers, count_wt, bin_widths[0])
print(p_value, z_value)


ax = plt.subplot(2,1,1, polar=True)

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
ax.set_title('WT', fontsize=12)
bars = ax.bar(bin_centers, count_wt, width=bin_widths, bottom=0.0, linewidth=0, edgecolor='black',alpha=0.8)
#ax.text(2,2, "Z = {:.1e}, P = {:.1e}".format(z_value, p_value), fontsize=7)


ax = plt.subplot(2,1,2, polar=True)

p_value, z_value = pycircstat.rayleigh(bin_centers, count_tg, bin_widths[0])
print(p_value, z_value)

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
ax.set_title('TG', fontsize=12)
bars = ax.bar(bin_centers, count_tg, width=bin_widths, bottom=0.0, linewidth=0, edgecolor='black',alpha=0.8)
#ax.text(2,2, "Z = {:.1e}, P = {:.1e}".format(z_value, p_value), fontsize=7)
plt.subplots_adjust(hspace=0)
plt.tight_layout()
plt.savefig('output/phase_modulation_analysis/spike_phase_int.png', dpi=300)"""


"""count_wt_pyr, _ = np.histogram(wt_phases_pyr, bins)
count_wt_int, _ = np.histogram(wt_phases_int, bins)

count_tg_pyr, _ = np.histogram(tg_phases_pyr, bins)
count_tg_int, _ = np.histogram(tg_phases_int, bins)



ax = plt.subplot(2,1,1, polar=True)

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
ax.set_title('WT', fontsize=12)
ax.plot(np.append(bin_centers, bin_centers[0]), np.append(count_wt_pyr, count_wt_pyr[0]))
ax.plot(np.append(bin_centers, bin_centers[0]), np.append(count_wt_int, count_wt_int[0]))

ax.vlines([pycircstat.mean(wt_phases_pyr), pycircstat.mean(wt_phases_int)], 0, [ax.get_ylim()[1], ax.get_ylim()[1]],colors=['C0', 'C1'])


#bars = ax.bar(bin_centers + bin_widths/4, count_wt_pyr, width=bin_widths/2, bottom=0.0, linewidth=0, edgecolor='black',alpha=0.8)
#bars = ax.bar(bin_centers - bin_widths/4, count_wt_int, width=bin_widths/2, bottom=0.0, linewidth=0, edgecolor='black',alpha=0.8)

#ax.text(2,2, "Z = {:.1e}, P = {:.1e}".format(z_value, p_value), fontsize=7)


ax = plt.subplot(2,1,2, polar=True)



plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
ax.set_title('TG', fontsize=12)
ax.plot(np.append(bin_centers, bin_centers[0]), np.append(count_tg_pyr, count_tg_pyr[0]))
ax.plot(np.append(bin_centers, bin_centers[0]), np.append(count_tg_int, count_tg_int[0]))
ax.vlines([pycircstat.mean(tg_phases_pyr), pycircstat.mean(tg_phases_int)], 0, [ax.get_ylim()[1], ax.get_ylim()[1]], colors=['C0', 'C1'])

plt.subplots_adjust(hspace=0)
plt.tight_layout()
plt.savefig('output/phase_modulation_analysis/spike_phase_pyr_int.png', dpi=300)"""

"""wt_locked_units = phase_df[(phase_df['sig'] == True) & (phase_df['strain'] == 'WT') & ((phase_df['session'] == 'Hab Rec 1') | (phase_df['session'] == 'OLM Rec 1'))]
wt_pyr = wt_locked_units[wt_locked_units['class'] == "PYR"]
wt_int = wt_locked_units[wt_locked_units['class'] == "INT"]

# get phase locked units
plt.figure(figsize=(5,8))
plt.rcParams['font.size'] = '16'

#sns.set_style("whitegrid")
wt_locked_units = phase_df[(phase_df['sig'] == True) & (phase_df['strain'] == 'WT') & ((phase_df['session'] == 'Hab Rec 1') | (phase_df['session'] == 'OLM Rec 1'))]
wt_pyr = wt_locked_units[wt_locked_units['class'] == "PYR"]
wt_int = wt_locked_units[wt_locked_units['class'] == "INT"]

tg_locked_units = phase_df[(phase_df['sig'] == True) & (phase_df['strain'] == 'TG') & ((phase_df['session'] == 'Hab Rec 1') | (phase_df['session'] == 'OLM Rec 1'))]
tg_pyr = tg_locked_units[tg_locked_units['class'] == "PYR"]
tg_int = tg_locked_units[tg_locked_units['class'] == "INT"]

wt_pyr_phases = wt_pyr['mean'].to_numpy() * np.pi/180
wt_pyr_z = wt_pyr['z_value'].to_numpy()
wt_int_phases = wt_int['mean'].to_numpy() * np.pi/180
wt_int_z = wt_int['z_value'].to_numpy()

tg_pyr_phases = tg_pyr['mean'].to_numpy() * np.pi/180
tg_pyr_z = tg_pyr['z_value'].to_numpy()
tg_int_phases = tg_int['mean'].to_numpy() * np.pi/180
tg_int_z = tg_int['z_value'].to_numpy()

ax = plt.subplot(2,1,1, polar=True)

#plt.xticks(fontsize=7)
#plt.yticks(fontsize=7)
#ax.set_title('WT', fontsize=12)
ax.set_yticks([])
ax.set_xticks([0, np.pi/2, np.pi, 3 * np.pi/2])


# scatter plot of phase preference
ax.scatter(wt_pyr_phases, np.log(wt_pyr_z), label="PYR")
ax.scatter(wt_int_phases, np.log(wt_int_z), label="INT")
ax.legend(loc='upper left')

ax.vlines([pycircstat.mean(wt_pyr_phases, wt_pyr_z), pycircstat.mean(wt_int_phases, wt_int_z)], 0, [ax.get_ylim()[1], ax.get_ylim()[1]],colors=['C0', 'C1'])

#bars = ax.bar(bin_centers, count_wt, width=bin_widths, bottom=0.0, linewidth=0, edgecolor='black',alpha=0.8)
#ax.text(2,2, "Z = {:.1e}, P = {:.1e}".format(z_value, p_value), fontsize=7)

ylim = ax.get_ylim()

ax = plt.subplot(2,1,2, polar=True)


#plt.xticks(fontsize=7)
#plt.yticks(fontsize=7)
#ax.set_title('TG', fontsize=12)
ax.set_yticks([])
ax.set_xticks([0, np.pi/2, np.pi, 3 * np.pi/2])


ax.scatter(tg_pyr_phases, np.log(tg_pyr_z))
ax.scatter(tg_int_phases, np.log(tg_int_z))
ax.vlines([pycircstat.mean(tg_pyr_phases, tg_pyr_z), pycircstat.mean(tg_int_phases, tg_int_z)], 0, [ax.get_ylim()[1], ax.get_ylim()[1]],colors=['C0', 'C1'])
ax.set_ylim(ylim)
#bars = ax.bar(bin_centers, count_tg, width=bin_widths, bottom=0.0, linewidth=0, edgecolor='black',alpha=0.8)
#ax.text(2,2, "Z = {:.1e}, P = {:.1e}".format(z_value, p_value), fontsize=7)
plt.subplots_adjust(hspace=0)
plt.tight_layout()
plt.savefig('output/phase_modulation_analysis/spike_phase_units.png', dpi=300)"""





"""plt.figure(figsize=(18, 8))

wt_locked_units = phase_df[(phase_df['sig'] == True) & (phase_df['strain'] == 'WT')]
tg_locked_units = phase_df[(phase_df['sig'] == True) & (phase_df['strain'] == 'TG')]

for i in range(0, 5):

    session = sessions[i]
    wt_pyr = wt_locked_units[(wt_locked_units['class'] == "PYR") & (wt_locked_units['session'] == session)]
    wt_int = wt_locked_units[(wt_locked_units['class'] == "INT") & (wt_locked_units['session'] == session)]

    tg_pyr = tg_locked_units[(tg_locked_units['class'] == "PYR") & (tg_locked_units['session'] == session)]
    tg_int = tg_locked_units[(tg_locked_units['class'] == "INT") & (tg_locked_units['session'] == session)]

    wt_pyr_phases = wt_pyr['mean'].to_numpy() * np.pi / 180
    wt_pyr_z = wt_pyr['z_value'].to_numpy()
    wt_int_phases = wt_int['mean'].to_numpy() * np.pi / 180
    wt_int_z = wt_int['z_value'].to_numpy()

    tg_pyr_phases = tg_pyr['mean'].to_numpy() * np.pi / 180
    tg_pyr_z = tg_pyr['z_value'].to_numpy()
    tg_int_phases = tg_int['mean'].to_numpy() * np.pi / 180
    tg_int_z = tg_int['z_value'].to_numpy()

    all_spike_phases = np.concatenate((np.concatenate(pyr_phases[i - 1]), np.concatenate(int_phases[i - 1])))

    count, _ = np.histogram(all_spike_phases, bins)
    ax = plt.subplot(2,5,i+1, polar=True)

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    ax.set_title(sessions[i] + ' ' + strains[i], fontsize=12)


    ax.scatter(wt_pyr_phases, np.log(wt_pyr_z), label="PYR")
    ax.scatter(wt_int_phases, np.log(wt_int_z), label="INT")
    if i == 0:
        ax.legend(loc="upper left")
    ax.vlines([pycircstat.mean(wt_phases_pyr), pycircstat.mean(wt_phases_int)], 0, [ax.get_ylim()[1], ax.get_ylim()[1]],
              colors=['C0', 'C1'])
    ylim = ax.get_ylim()
    ax.set_yticks([])
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])




    ax = plt.subplot(2,5,i+6, polar=True)
    ax.set_title(sessions[i+5] + ' ' + strains[i+5], fontsize=12)

    ax.scatter(tg_pyr_phases, np.log(tg_pyr_z))
    ax.scatter(tg_int_phases, np.log(tg_int_z))
    ax.vlines([pycircstat.mean(tg_pyr_phases, tg_pyr_z), pycircstat.mean(tg_int_phases, tg_int_z)], 0,
              [ax.get_ylim()[1], ax.get_ylim()[1]], colors=['C0', 'C1'])
    ax.set_ylim(ylim)
    ax.set_yticks([])
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
plt.subplots_adjust(hspace=0)
#ax.set_axisbelow(False)
# make plot pretty
# reduce font size
plt.tight_layout()
plt.savefig('output/phase_modulation_analysis/spike_phase_units_session.png', dpi=300)
"""


# Rose plot
"""plt.figure(figsize=(18, 8))



for i in range(1, len(pyr_phases) + 1):


    all_spike_phases = np.concatenate((np.concatenate(pyr_phases[i - 1]), np.concatenate(int_phases[i - 1])))

    count, _ = np.histogram(all_spike_phases, bins)
    ax = plt.subplot(2,5,i, polar=True)

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    ax.set_title(sessions[i-1] + ' ' + strains[i-1], fontsize=12)
    bars = ax.bar(bin_centers, count, width=bin_widths, bottom=0.0, linewidth=0, edgecolor='black',alpha=0.8)
plt.subplots_adjust(hspace=0)
#ax.set_axisbelow(False)
# make plot pretty
# reduce font size
plt.tight_layout()
plt.savefig('output/phase_modulation_analysis/spike_phase.png', dpi=300)"""


"""plt.figure(figsize=(18, 8))

for i in range(1, len(pyr_phases) + 1):


    all_spike_phases = np.concatenate(pyr_phases[i - 1])

    count, _ = np.histogram(all_spike_phases, bins)
    ax = plt.subplot(2,5,i, polar=True)

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    ax.set_title(sessions[i-1] + ' ' + strains[i-1], fontsize=12)
    bars = ax.bar(bin_centers, count, width=bin_widths, bottom=0.0, linewidth=0, edgecolor='black',alpha=0.8)
plt.subplots_adjust(hspace=0)
#ax.set_axisbelow(False)
# make plot pretty
# reduce font size
plt.tight_layout()
plt.savefig('output/phase_modulation_analysis/spike_phase_session_pyr.png', dpi=300)"""


"""plt.figure(figsize=(18, 8))

for i in range(1, len(int_phases) + 1):


    all_spike_phases = np.concatenate(int_phases[i - 1])

    count, _ = np.histogram(all_spike_phases, bins)
    ax = plt.subplot(2,5,i, polar=True)

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    ax.set_title(sessions[i-1] + ' ' + strains[i-1], fontsize=12)
    bars = ax.bar(bin_centers, count, width=bin_widths, bottom=0.0, linewidth=0, edgecolor='black',alpha=0.8,color='C1')
plt.subplots_adjust(hspace=0)
#ax.set_axisbelow(False)
# make plot pretty
# reduce font size
plt.tight_layout()
plt.savefig('output/phase_modulation_analysis/spike_phase_session_int.png', dpi=300)"""

# count number and proportion that are significant


"""plt.figure(figsize=(18, 8))

for i in range(1, len(pyr_phases) + 1):

    count_pyr, _ = np.histogram(np.concatenate(pyr_phases[i - 1]), bins)
    count_int, _ = np.histogram(np.concatenate(int_phases[i - 1]), bins)
    ax = plt.subplot(2,5,i, polar=True)

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    ax.set_title(sessions[i-1] + ' ' + strains[i-1], fontsize=12)
    ax.plot(np.append(bin_centers, bin_centers[0]), np.append(count_pyr, count_pyr[0]))
    ax.plot(np.append(bin_centers, bin_centers[0]), np.append(count_int, count_int[0]))

    ax.vlines([pycircstat.mean(np.concatenate(pyr_phases[i - 1])), pycircstat.mean(np.concatenate(int_phases[i - 1]))], 0, [ax.get_ylim()[1], ax.get_ylim()[1]],
              colors=['C0', 'C1'])
plt.subplots_adjust(hspace=0)
#ax.set_axisbelow(False)
# make plot pretty
# reduce font size
plt.tight_layout()
plt.savefig('output/phase_modulation_analysis/spike_phase_session_combined.png', dpi=300)"""




"""
2. Compare phase locking strength distributions

"""


custom_long_name = 'KS 2 samp statistical test '
custom_short_name = 'KS'
custom_func = stats.ks_2samp
custom_test = StatTest(custom_func, custom_long_name, custom_short_name)

"""wt = phase_df[phase_df['strain'] == 'WT']
tg = phase_df[phase_df['strain'] == 'TG']
# compare across all units
print(stats.ks_2samp(wt['z_value'].to_numpy(), tg['z_value'].to_numpy()))"""



"""sns.ecdfplot(phase_df, x="z_value", hue="strain")
plt.xlabel("Strength of phase locking (log z value)")
plt.title("CDF of ripple phase locking strength")
plt.tight_layout
plt.savefig('output/all_z_value_cdf.png', dpi=300)"""



"""ax = sns.violinplot(data=phase_df, x="strain", y="z_value", legend=False)
annot = Annotator(ax, [("WT", "TG")], data=phase_df, x='strain', y='z_value')
annot.configure
annot.configure(test=custom_test, comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
plt.xlabel("Strain")
plt.ylabel("Strength of phase locking (log z value)")
plt.title("Phase locking strength", y=1.08)

#plt.tight_layout()
#plt.show()
plt.savefig('output/all_violin_phase_lock.png', dpi=300)"""


# KS significance but only with phase locked units

"""phase_df_sig = phase_df[phase_df['sig'] == True]
pyr_phase_df = phase_df_sig[phase_df['class'] == 'PYR']
int_phase_df = phase_df_sig[phase_df['class'] == 'INT']
phase_df_sig['Condition'] = phase_df_sig.apply(lambda row: row['strain'] + " " + row['class'], axis=1)

wt = phase_df_sig[(phase_df['strain'] == 'WT')]
tg = phase_df_sig[(phase_df['strain'] == 'TG')]

print(wt['z_value'].to_numpy().size)
print(stats.ks_2samp(tg['z_value'].to_numpy(), wt['z_value'].to_numpy()))
"""

"""sns.ecdfplot(phase_df_sig, x="z_value", hue="strain")
plt.xlabel("Strength of phase locking (log z value)")
plt.title("CDF of ripple phase locking strength (phase locked units)")
plt.tight_layout
plt.savefig('output/phase_modulation_analysis/z_cdf.png', dpi=300)"""


"""plt.figure(figsize=(6,8))
phase_df_sig = phase_df_sig[(phase_df['session'] != 'OLM Rec 3')]
plt.rcParams['font.size'] = '16'

ax = sns.ecdfplot(phase_df_sig, x="z_value", hue="Condition")
plt.xlabel("Strength of phase locking (log z value)")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.tight_layout
plt.savefig('output/phase_modulation_analysis/z_cdf_class.png', dpi=300)"""


"""ax = sns.violinplot(data=phase_df_sig, x="strain", y="z_value", legend=False)
annot = Annotator(ax, [("WT", "TG")], data=phase_df, x='strain', y='z_value')
annot.configure
annot.configure(test=custom_test, comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
plt.xlabel("Strain")
plt.ylabel("Strength of phase locking (log z value)")
plt.title("Phase locking strength (phase locked units)", y=1.08)
plt.savefig('output/phase_modulation_analysis/z_violin.png', dpi=300)"""

"""phase_df_sig = phase_df_sig[(phase_df['session'] != 'OLM Rec 3')]

plt.figure(figsize=(6,8))
plt.rcParams['font.size'] = '16'

ax = sns.violinplot(data=phase_df_sig, x="Condition", y="z_value", legend=False)
annot = Annotator(ax, [("WT PYR", "TG PYR"), ("WT INT", "TG INT")], data=phase_df_sig, x='Condition', y='z_value')
annot.configure
annot.configure(test=custom_test, comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()

plt.xlabel("")
plt.ylabel("Strength of phase locking (log z value)")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.savefig('output/phase_modulation_analysis/z_violin_class.png', dpi=300)"""

# KS test between conditions
# WT only

"""wt = phase_df[(phase_df['strain'] == 'WT') & (phase_df['sig'] == True)]
wt_1 = phase_df[(phase_df['strain'] == 'WT') & (phase_df['sig'] == True) & (phase_df['session'] == "Hab Rec 1")]
wt_2 = phase_df[(phase_df['strain'] == 'WT') & (phase_df['sig'] == True) & (phase_df['session'] == "Hab Rec 2")]
wt_3 = phase_df[(phase_df['strain'] == 'WT') & (phase_df['sig'] == True) & (phase_df['session'] == "OLM Rec 1")]
wt_4 = phase_df[(phase_df['strain'] == 'WT') & (phase_df['sig'] == True) & (phase_df['session'] == "OLM Rec 2")]
wt_5 = phase_df[(phase_df['strain'] == 'WT') & (phase_df['sig'] == True) & (phase_df['session'] == "OLM Rec 3")]

print(stats.ks_2samp(wt_1['z_value'].to_numpy(), wt_2['z_value'].to_numpy()))
print(stats.ks_2samp(wt_1['z_value'].to_numpy(), wt_3['z_value'].to_numpy()))

print(stats.ks_2samp(wt_3['z_value'].to_numpy(), wt_4['z_value'].to_numpy()))
print(stats.ks_2samp(wt_3['z_value'].to_numpy(), wt_5['z_value'].to_numpy()))
print(stats.ks_2samp(wt_4['z_value'].to_numpy(), wt_5['z_value'].to_numpy()))"""

"""plt.figure(figsize=(5,8))
ax = sns.ecdfplot(wt, x="z_value", hue="session")
#sns.violinplot(data=wt, x="session", y="z_value")
plt.xlabel("Strength of phase locking (log z value)")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.tight_layout()

#plt.show()
plt.savefig('output/phase_modulation_analysis/wt_cdf.png', dpi=300)"""



"""ax = sns.violinplot(data=wt, x="session", y="z_value", legend=False)
annot = Annotator(ax, [("Hab Rec 1", "Hab Rec 2"), ("OLM Rec 1", "OLM Rec 2"), ("OLM Rec 1", "OLM Rec 3")], data=wt, x="session", y="z_value")
annot.configure
annot.configure(test=custom_test, comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
plt.xlabel("")
plt.ylabel("Strength of phase locking (log z value)")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.tight_layout()
#plt.show()
plt.savefig('output/phase_modulation_analysis/wt_violin_ks_test.png', dpi=300)"""

"""plt.figure(figsize=(5,8))
ax = sns.violinplot(data=wt, x="session", y="z_value", legend=False)
annot = Annotator(ax, [("Hab Rec 1", "Hab Rec 2"), ("OLM Rec 1", "OLM Rec 2"), ("OLM Rec 1", "OLM Rec 3")], data=wt, x="session", y="z_value")
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction="Bonferroni",
                text_format='star', loc="outside").apply_test().annotate()
plt.xlabel("")
plt.ylabel("Strength of phase locking (log z value)")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.tight_layout()
#plt.show()
plt.savefig('output/phase_modulation_analysis/wt_violin.png', dpi=300)"""

"""plt.figure(figsize=(5,8))
pairs = [
    (("Hab Rec 1", "PYR"), ("Hab Rec 2", "PYR")),
    (("Hab Rec 1", "INT"), ("Hab Rec 2", "INT")),
    (("OLM Rec 1", "PYR"), ("OLM Rec 2", "PYR")),
    (("OLM Rec 1", "INT"), ("OLM Rec 2", "INT")),
    (("OLM Rec 1", "PYR"), ("OLM Rec 3", "PYR")),
    (("OLM Rec 1", "INT"), ("OLM Rec 3", "INT")),

]
ax = sns.violinplot(data=wt, x="session", y="z_value", hue="class", legend=False)
annot = Annotator(ax, pairs, data=wt, x="session", y="z_value", hue="class")
annot.configure
annot.configure(test=custom_test, comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
plt.xlabel("")
plt.ylabel("Strength of phase locking (z value)")
#plt.title("Phase locking strength (WT)", y=1.08)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.tight_layout()
#plt.show()
plt.savefig('output/phase_modulation_analysis/wt_violin_class.png', dpi=300)"""


# TG only
"""tg = phase_df[(phase_df['strain'] == 'TG') & (phase_df['sig'] == True)]
tg_1 = phase_df[(phase_df['strain'] == 'TG') & (phase_df['sig'] == True) & (phase_df['session'] == "Hab Rec 1")]
tg_2 = phase_df[(phase_df['strain'] == 'TG') & (phase_df['sig'] == True) & (phase_df['session'] == "Hab Rec 2")]
tg_3 = phase_df[(phase_df['strain'] == 'TG') & (phase_df['sig'] == True) & (phase_df['session'] == "OLM Rec 1")]
tg_4 = phase_df[(phase_df['strain'] == 'TG') & (phase_df['sig'] == True) & (phase_df['session'] == "OLM Rec 2")]
tg_5 = phase_df[(phase_df['strain'] == 'TG') & (phase_df['sig'] == True) & (phase_df['session'] == "OLM Rec 3")]

print(stats.ks_2samp(tg_1['z_value'].to_numpy(), tg_2['z_value'].to_numpy()))
print(stats.ks_2samp(tg_1['z_value'].to_numpy(), tg_3['z_value'].to_numpy()))

print(stats.ks_2samp(tg_3['z_value'].to_numpy(), tg_4['z_value'].to_numpy()))
print(stats.ks_2samp(tg_3['z_value'].to_numpy(), tg_5['z_value'].to_numpy()))
print(stats.ks_2samp(tg_4['z_value'].to_numpy(), tg_5['z_value'].to_numpy()))"""



"""plt.figure(figsize=(5,8))
ax = sns.ecdfplot(tg, x="z_value", hue="session")
#sns.violinplot(data=wt, x="session", y="z_value")
plt.xlabel("Strength of phase locking (log z value)")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.tight_layout()
#plt.show()
plt.savefig('output/phase_modulation_analysis/tg_cdf.png', dpi=300)"""

"""plt.figure(figsize=(5,8))
ax = sns.violinplot(data=tg, x="session", y="z_value", legend=False)
annot = Annotator(ax, [("Hab Rec 1", "Hab Rec 2"), ("OLM Rec 1", "OLM Rec 2"), ("OLM Rec 1", "OLM Rec 3")], data=tg, x="session", y="z_value")
annot.configure
annot.configure(test=custom_test, comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
plt.xlabel("")
plt.ylabel("Strength of phase locking (z value)")

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.tight_layout()
#plt.show()
plt.savefig('output/phase_modulation_analysis/tg_violin.png', dpi=300)"""

"""plt.figure(figsize=(5,8))
ax = sns.violinplot(data=tg, x="session", y="z_value", hue="class", legend=False)
pairs = [
    (("Hab Rec 1", "PYR"), ("Hab Rec 2", "PYR")),
    (("Hab Rec 1", "INT"), ("Hab Rec 2", "INT")),
    (("OLM Rec 1", "PYR"), ("OLM Rec 2", "PYR")),
    (("OLM Rec 1", "INT"), ("OLM Rec 2", "INT")),
    (("OLM Rec 1", "PYR"), ("OLM Rec 3", "PYR")),
    (("OLM Rec 1", "INT"), ("OLM Rec 3", "INT")),

]
annot = Annotator(ax, pairs, data=tg, x="session", y="z_value", hue="class")
annot.configure
annot.configure(test=custom_test, comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
plt.xlabel("")
plt.ylabel("Strength of phase locking (z value)")
#plt.title("Phase locking strength (TG)")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.tight_layout()
#plt.show()
plt.savefig('output/phase_modulation_analysis/tg_violin_class.png', dpi=300)"""

# num units


"""ax = sns.catplot(kind='bar',x='session', y='count', hue='strain', data=sig_units_df, legend_out=True, ci='sd', errwidth=1.0, capsize=0.2)
ax.set(xlabel='Recording Sessions', ylabel='Percent', title='Percent of units with significant phase locking to ripples', ylim=(0,100))

ax._legend.set_title('Strain')
plt.tight_layout()
plt.savefig(fname='output/percent_sig_phase_bonferroni.png',bbox_inches='tight', dpi=300)"""




#print(phase_df[phase_df['p_value'] < 0.05].index)

#plt.figure()
#sns.displot(phase_df, x='z_value', hue='strain')
#plt.show()


# look at distributions of medians
# WT vs TG









# look at single units instead of total activity
# TODO: generate a distribution for this (use von mises)
"""wt = phase_df[phase_df['strain'] == 'WT']
tg = phase_df[phase_df['strain'] == 'TG']
# compare across all units
print(stats.ks_2samp(wt['mean'].to_numpy(), tg['mean'].to_numpy()))
sns.histplot(phase_df, x="mean", hue="strain", bins=32, multiple="stack")
plt.ylabel("Phase")
plt.title("Distribution of mean ripple spiking phases")
plt.savefig('output/mean_phase_dist.png', dpi=300)"""


"""phase_df_sig = phase_df[phase_df['sig'] == True]
wt = phase_df[(phase_df['strain'] == 'WT') & (phase_df['sig'] == True)]
tg = phase_df[(phase_df['strain'] == 'TG') & (phase_df['sig'] == True)]

print(stats.ks_2samp(tg['mean'].to_numpy(), wt['mean'].to_numpy()))

ax = sns.histplot(phase_df_sig, x="mean", hue="strain", bins=32, multiple="stack")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.ylabel("Phase")
plt.savefig('output/phase_modulation_analysis/mean_phase_dist_sig.png', dpi=300)"""

"""plt.figure()
plt.rcParams['font.size'] = '14'

#((phase_df['session'] == 'Hab Rec 1') | (phase_df['session'] == 'OLM Rec 1'))

phase_df_sig = phase_df[(phase_df['sig'] == True) & (phase_df['class'] == "PYR") & ((phase_df['session'] == 'Hab Rec 1') | (phase_df['session'] == 'OLM Rec 1'))]
wt = phase_df[(phase_df['strain'] == 'WT') & (phase_df['sig'] == True) & (phase_df['class'] == "PYR")]
tg = phase_df[(phase_df['strain'] == 'TG') & (phase_df['sig'] == True) & (phase_df['class'] == "PYR")]

print(stats.ks_2samp(tg['mean'].to_numpy(), wt['mean'].to_numpy()))

ax = sns.histplot(phase_df_sig, x="mean", hue="strain", bins=32, multiple="stack")
plt.xticks([-180,-90,0,90,180])
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.ylabel("Count")
plt.xlabel("Phase (deg)")
plt.savefig('output/phase_modulation_analysis/mean_phase_dist_sig_pyr.png', dpi=300)

plt.figure()
plt.rcParams['font.size'] = '14'

phase_df_sig = phase_df[(phase_df['sig'] == True) & (phase_df['class'] == "INT") & ((phase_df['session'] == 'Hab Rec 1') | (phase_df['session'] == 'OLM Rec 1'))]
wt = phase_df[(phase_df['strain'] == 'WT') & (phase_df['sig'] == True) & (phase_df['class'] == "INT")]
tg = phase_df[(phase_df['strain'] == 'TG') & (phase_df['sig'] == True) & (phase_df['class'] == "INT")]

print(stats.ks_2samp(tg['mean'].to_numpy(), wt['mean'].to_numpy()))

ax = sns.histplot(phase_df_sig, x="mean", hue="strain", bins=32, multiple="stack")
plt.xticks([-180,-90,0,90,180])
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.ylabel("Count")
plt.xlabel("Phase (deg)")
plt.savefig('output/phase_modulation_analysis/mean_phase_dist_sig_int.png', dpi=300)"""
