import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
import scipy.stats as stats

custom_long_name = 'KS 2 samp statistical test '
custom_short_name = 'KS'
custom_func = stats.ks_2samp
custom_test = StatTest(custom_func, custom_long_name, custom_short_name)
plt.rcParams['font.size'] = '16'
phase_df = pd.read_csv('../output/phase_info_hour.csv')
#phase_df = pd.read_csv('../output/phase_info_ripple_limited.csv')
#sig_units_df = pd.read_csv('../output/sig_phase_info.csv')




pyr_df = phase_df[(phase_df['session'] == "Sleep1") | (phase_df['session'] == "Sleep2")]
pyr_df['z_value'] = pyr_df.apply(lambda row: np.log(row['z_value']), axis=1)
pyr_df = pyr_df[pyr_df['sig'] == True]
pyr_df = pyr_df[(pyr_df['task'] == "hab")]
pyr_df = pyr_df[pyr_df['class'] == 'PYR']
pyr_df = pyr_df[pyr_df['num_spikes'] > 50]

ax = sns.violinplot(data=pyr_df, x="strain", y="z_value", hue='session', split=True)
sns.swarmplot(data=pyr_df, x="strain", y="z_value", hue='session',dodge=True)

annot = Annotator(ax, [(('wt', 'Sleep1'), ('wt', 'Sleep2')), (('tg', 'Sleep1'), ('tg', 'Sleep2'))], data=pyr_df, x="strain", y="z_value", hue='session')
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
#sns.violinplot(data=wt, x="session", y="z_value")
#plt.ylabel("mean phase")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.tight_layout()
plt.show()


pyr_df = phase_df[(phase_df['session'] == "Sleep1") | (phase_df['session'] == "Sleep2")]
pyr_df['z_value'] = pyr_df.apply(lambda row: np.log(row['z_value']), axis=1)
pyr_df = pyr_df[pyr_df['sig'] == True]
pyr_df = pyr_df[(pyr_df['task'] == "olm")]
pyr_df = pyr_df[pyr_df['class'] == 'PYR']
pyr_df = pyr_df[pyr_df['num_spikes'] > 50]

ax = sns.violinplot(data=pyr_df, x="strain", y="z_value", hue='session', split=True)
sns.swarmplot(data=pyr_df, x="strain", y="z_value", hue='session',dodge=True)

annot = Annotator(ax, [(('wt', 'Sleep1'), ('wt', 'Sleep2')), (('tg', 'Sleep1'), ('tg', 'Sleep2'))], data=pyr_df, x="strain", y="z_value", hue='session')
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
#sns.violinplot(data=wt, x="session", y="z_value")
#plt.ylabel("mean phase")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.tight_layout()
plt.show()








df1 = phase_df[(phase_df['session'] == "Sleep1")]
df2 = phase_df[(phase_df['session'] == "Sleep2")]
df1 = df1.reset_index(drop=True)
df2 = df2.reset_index(drop=True)


df1['z_value_post'] = df2['z_value']
df1['ppc_post'] = df2['ppc']
df1['num_spikes_post'] = df2['num_spikes']
df1['sig_post'] = df2['sig']

df = df1
df[df['class'] == 'PYR']
print(len(df[df['sig'] == True]))
print(len(df[df['sig_post'] == True]))
df['sig'] = df.apply(lambda row: row['sig'] | row['sig_post'], axis=1)
df['relative_z'] = df.apply(lambda row: np.log(row['z_value_post']) - np.log(row['z_value']), axis=1)
df['delta_ppc'] = df.apply(lambda row: row['ppc_post'] - row['ppc'], axis=1)


df = df[df['sig_post'] == True]


pyr_df = df[df['class'] == 'PYR']
pyr_df = pyr_df[pyr_df['num_spikes'] > 50]
pyr_df = pyr_df[pyr_df['num_spikes_post'] > 50]


plt.figure(figsize=(8,5))
#wt_pyr_df = pyr_df[pyr_df["strain"] == "WT"]
#print(stats.wilcoxon(wt_pyr_df[wt_pyr_df["Condition"] == "OLM"]["diff"]))
#print(stats.wilcoxon(wt_pyr_df[wt_pyr_df["Condition"] == "Hab"]["diff"]))





# compare phase locking between sleep 2



ax = sns.violinplot(data=pyr_df, x="strain", y="relative_z", hue='task', split=True)

annot = Annotator(ax, [(('wt', 'hab'), ('wt', 'olm')), (('tg', 'hab'), ('tg', 'olm'))], data=pyr_df, x="strain", y="relative_z", hue='task')
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
#sns.violinplot(data=wt, x="session", y="z_value")
#plt.ylabel("mean phase")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.tight_layout()
plt.show()
#plt.savefig('../output/phase_modulation_analysis/task_phase_strength/task_violin_wt_pyr_mean.png', dpi=300)
















import pandas as pd
import pycircstat
import matplotlib.pyplot as plt
import numpy as np
import pycircstat
import unit_utils
import os
import seaborn as sns
import scipy.io as sio
import kuiper
import pickle
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
#sns.set_style('ticks')

plt.figure()
plt.rcParams['font.size'] = '14'
plt.rcParams['axes.linewidth'] = 2


phase_df = pd.read_csv('../output/phase_info_hour.csv')
pyr_df = phase_df[(phase_df['session'] == "Sleep1") | (phase_df['session'] == "Sleep2")]
phase_df_sig = pyr_df[(pyr_df['sig'] == True) & (pyr_df['class'] == "PYR") & (pyr_df['task'] == 'hab')]
wt = phase_df_sig[(phase_df_sig['strain'] == 'wt')]
tg = phase_df_sig[(phase_df_sig['strain'] == 'tg')]

print(len(wt[wt['session'] == 'Sleep1']))
print(len(wt[wt['session'] == 'Sleep2']))
print(kuiper.kuiper_two(wt['mean'].to_numpy(), tg['mean'].to_numpy()))

#print(stats.ks_2samp(tg['mean'].to_numpy(), wt['mean'].to_numpy()))
bins = np.linspace(-180, 180, 33)
bin_width = (bins[1] - bins[0])
ax = sns.histplot(wt, x="mean", hue="session", binwidth=bin_width, binrange=(bins[0],bins[-1]), linewidth=0.5)
plt.xticks([-180,-90,0,90,180])
ax.set_xlim(-180, 180)
#ax.text(-15,35,"****")
#ax.set_ylim(0, 15)
ax.tick_params(width=2)
#ax.legend(title="",frameon=False)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.ylabel("Count")
plt.xlabel("Phase (deg)")
plt.show()
#plt.savefig('../output/phase_modulation_analysis/mean_phase_dist_sig_pyr.png', dpi=300)




plt.figure()
plt.rcParams['font.size'] = '14'
plt.rcParams['axes.linewidth'] = 2


phase_df = pd.read_csv('../output/phase_info_hour.csv')
pyr_df = phase_df[(phase_df['session'] == "Sleep1") | (phase_df['session'] == "Sleep2")]
phase_df_sig = pyr_df[(pyr_df['sig'] == True) & (pyr_df['class'] == "PYR") & (pyr_df['task'] == 'olm')]
wt = phase_df_sig[(phase_df_sig['strain'] == 'wt')]
tg = phase_df_sig[(phase_df_sig['strain'] == 'tg')]


print(kuiper.kuiper_two(wt['mean'].to_numpy(), tg['mean'].to_numpy()))

#print(stats.ks_2samp(tg['mean'].to_numpy(), wt['mean'].to_numpy()))
bins = np.linspace(-180, 180, 33)
bin_width = (bins[1] - bins[0])
ax = sns.histplot(wt, x="mean", hue="session", binwidth=bin_width, binrange=(bins[0],bins[-1]), linewidth=0.5)
plt.xticks([-180,-90,0,90,180])
ax.set_xlim(-180, 180)
#ax.text(-15,35,"****")
#ax.set_ylim(0, 15)
ax.tick_params(width=2)
#ax.legend(title="",frameon=False)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.ylabel("Count")
plt.xlabel("Phase (deg)")
plt.show()



phase_df = pd.read_csv('../output/phase_info_hour.csv')
pyr_df = phase_df[(phase_df['session'] == "Sleep1") | (phase_df['session'] == "Sleep2")]
phase_df_sig = pyr_df[(pyr_df['sig'] == True) & (pyr_df['class'] == "PYR") & (pyr_df['task'] == 'hab')]
wt = phase_df_sig[(phase_df_sig['strain'] == 'wt')]
tg = phase_df_sig[(phase_df_sig['strain'] == 'tg')]


print(kuiper.kuiper_two(wt['mean'].to_numpy(), tg['mean'].to_numpy()))


#print(stats.ks_2samp(tg['mean'].to_numpy(), wt['mean'].to_numpy()))
bins = np.linspace(-180, 180, 33)
bin_width = (bins[1] - bins[0])
ax = sns.histplot(tg, x="mean", hue="session", binwidth=bin_width, binrange=(bins[0],bins[-1]), linewidth=0.5)
plt.xticks([-180,-90,0,90,180])
ax.set_xlim(-180, 180)
#ax.text(-15,35,"****")
#ax.set_ylim(0, 15)
ax.tick_params(width=2)
#ax.legend(title="",frameon=False)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.ylabel("Count")
plt.xlabel("Phase (deg)")
plt.show()


phase_df = pd.read_csv('../output/phase_info_hour.csv')
pyr_df = phase_df[(phase_df['session'] == "Sleep1") | (phase_df['session'] == "Sleep2")]
phase_df_sig = pyr_df[(pyr_df['sig'] == True) & (pyr_df['class'] == "PYR") & (pyr_df['task'] == 'olm')]
wt = phase_df_sig[(phase_df_sig['strain'] == 'wt')]
tg = phase_df_sig[(phase_df_sig['strain'] == 'tg')]

ax = sns.histplot(tg, x="mean", hue="session", binwidth=bin_width, binrange=(bins[0],bins[-1]), linewidth=0.5)
plt.xticks([-180,-90,0,90,180])
ax.set_xlim(-180, 180)
#ax.text(-15,35,"****")
#ax.set_ylim(0, 15)
ax.tick_params(width=2)
#ax.legend(title="",frameon=False)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.ylabel("Count")
plt.xlabel("Phase (deg)")
plt.show()



