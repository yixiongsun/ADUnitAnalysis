import pycircstat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
phase_df = pd.read_csv('../output/phase_info.csv')
#sig_units_df = pd.read_csv('../output/sig_phase_info.csv')

plt.figure(figsize=(10,5))
plt.rcParams['font.size'] = '16'
#phase_df = phase_df[phase_df['id'] != '64-30']

#sns.set_style("whitegrid")
wt_locked_units = phase_df[(phase_df['sig'] == True) & (phase_df['strain'] == 'wt') & ((phase_df['session'] == 'combined'))]
wt_pyr = wt_locked_units[wt_locked_units['class'] == "PYR"]
wt_int = wt_locked_units[wt_locked_units['class'] == "INT"]

tg_locked_units = phase_df[(phase_df['sig'] == True) & (phase_df['strain'] == 'tg') & ((phase_df['session'] == 'combined'))]
tg_pyr = tg_locked_units[tg_locked_units['class'] == "PYR"]
tg_int = tg_locked_units[tg_locked_units['class'] == "INT"]

wt_pyr_phases = wt_pyr['mean'].to_numpy() * np.pi/180
wt_pyr_z = wt_pyr['z_value'].to_numpy()
wt_pyr_ppc = wt_pyr['ppc'].to_numpy()
wt_int_phases = wt_int['mean'].to_numpy() * np.pi/180
wt_int_z = wt_int['z_value'].to_numpy()
wt_int_ppc = wt_int['ppc'].to_numpy()

tg_pyr_phases = tg_pyr['mean'].to_numpy() * np.pi/180
tg_pyr_z = tg_pyr['z_value'].to_numpy()
tg_pyr_ppc = tg_pyr['ppc'].to_numpy()
tg_int_phases = tg_int['mean'].to_numpy() * np.pi/180
tg_int_z = tg_int['z_value'].to_numpy()
tg_int_ppc = tg_int['ppc'].to_numpy()

ax = plt.subplot(1,2,1, polar=True)

#plt.xticks(fontsize=7)
#plt.yticks(fontsize=7)
#ax.set_title('WT', fontsize=12)
ax.set_yticks([])
ax.set_xticks([0, np.pi/2, np.pi, 3 * np.pi/2])

ax.set_rticks([6], ['log(z)=6'])
ax.scatter(wt_pyr_phases, np.log(wt_pyr_z), marker='.', label="WT", color=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765))
ax.scatter(tg_pyr_phases, np.log(tg_pyr_z), marker='.', label='TG', color=(1.0, 0.4980392156862745, 0.054901960784313725))


print(ax.get_rmax())


#ax.scatter(wt_pyr_phases, wt_pyr_ppc, label="PYR")
#ax.scatter(wt_int_phases, wt_int_ppc, label="INT")
ax.legend(loc=(-0.3, 0.9), frameon=False)

#ax.vlines([pycircstat.mean(wt_pyr_phases, wt_pyr_z), pycircstat.mean(wt_int_phases, wt_int_z)], 0, [ax.get_ylim()[1], ax.get_ylim()[1]],colors=['C0', 'C1'])
ax.vlines([pycircstat.mean(wt_pyr_phases), pycircstat.mean(tg_pyr_phases)], 0, [6, 6],colors=[(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (1.0, 0.4980392156862745, 0.054901960784313725)])
#bars = ax.bar(bin_centers, count_wt, width=bin_widths, bottom=0.0, linewidth=0, edgecolor='black',alpha=0.8)
#ax.text(2,2, "Z = {:.1e}, P = {:.1e}".format(z_value, p_value), fontsize=7)
ax.set_rmax(6)
#ylim = ax.get_ylim()

ax = plt.subplot(1,2,2, polar=True)


#plt.xticks(fontsize=7)
#plt.yticks(fontsize=7)
#ax.set_title('TG', fontsize=12)
ax.set_yticks([])
ax.set_xticks([0, np.pi/2, np.pi, 3 * np.pi/2])
ax.set_rticks([6], ['log(z)=6'])
ax.scatter(wt_int_phases, np.log(wt_int_z), marker='.', color=(0.6509803921568628, 0.807843137254902, 0.8901960784313725))
ax.scatter(tg_int_phases, np.log(tg_int_z), marker='.', color=(0.9921568627450981, 0.7490196078431373, 0.43529411764705883))

#ax.vlines([pycircstat.mean(tg_pyr_phases, tg_pyr_z), pycircstat.mean(tg_int_phases, tg_int_z)], 0, [ax.get_ylim()[1], ax.get_ylim()[1]],colors=['C0', 'C1'])
ax.vlines([pycircstat.mean(wt_int_phases), pycircstat.mean(tg_int_phases)], 0, [6, 6], colors=[(0.6509803921568628, 0.807843137254902, 0.8901960784313725), (0.9921568627450981, 0.7490196078431373, 0.43529411764705883)])
#ax.set_ylim(ylim)
ax.set_rmax(6)
print(ax.get_rmax())
#bars = ax.bar(bin_centers, count_tg, width=bin_widths, bottom=0.0, linewidth=0, edgecolor='black',alpha=0.8)
#ax.text(2,2, "Z = {:.1e}, P = {:.1e}".format(z_value, p_value), fontsize=7)
plt.subplots_adjust(hspace=0)
plt.tight_layout()
#plt.show()
plt.savefig('../output/phase_modulation_analysis/spike_phase_units.png', dpi=300)