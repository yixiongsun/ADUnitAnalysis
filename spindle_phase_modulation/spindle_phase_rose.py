import pycircstat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

phase_df = pd.read_csv('../output/spindle_phase_info.csv')
sig_units_df = pd.read_csv('../output/spindle_sig_phase_info.csv')


# get phase locked units
plt.figure(figsize=(5,8))
plt.rcParams['font.size'] = '16'

#sns.set_style("whitegrid")
wt_locked_units = phase_df[(phase_df['sig'] == True) & (phase_df['strain'] == 'WT')]# & ((phase_df['session'] == 'Hab Rec 1') | (phase_df['session'] == 'OLM Rec 1'))]
wt_pyr = wt_locked_units[wt_locked_units['class'] == "PYR"]
wt_int = wt_locked_units[wt_locked_units['class'] == "INT"]

tg_locked_units = phase_df[(phase_df['sig'] == True) & (phase_df['strain'] == 'TG')]# & ((phase_df['session'] == 'Hab Rec 1') | (phase_df['session'] == 'OLM Rec 1'))]
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
plt.savefig('../output/spindle_phase_modulation_analysis/spindle_phase_rose.png', dpi=300)