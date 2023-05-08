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


phase_df = pd.read_csv('../output/phase_info.csv')
#((phase_df['session'] == 'Hab Rec 1') | (phase_df['session'] == 'OLM Rec 1'))

phase_df_sig = phase_df[(phase_df['sig'] == True) & (phase_df['class'] == "PYR") & (phase_df['session'] == 'combined')]
wt = phase_df[(phase_df['strain'] == 'wt') & (phase_df['sig'] == True) & (phase_df['class'] == "PYR")]
tg = phase_df[(phase_df['strain'] == 'tg') & (phase_df['sig'] == True) & (phase_df['class'] == "PYR")]


print(kuiper.kuiper_two(wt['mean'].to_numpy(), tg['mean'].to_numpy()))

#print(stats.ks_2samp(tg['mean'].to_numpy(), wt['mean'].to_numpy()))
bins = np.linspace(-180, 180, 33)
bin_width = (bins[1] - bins[0])
ax = sns.histplot(phase_df_sig, x="mean", hue="strain", binwidth=bin_width, binrange=(bins[0],bins[-1]), element="step",linewidth=1,palette=[(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (1.0, 0.4980392156862745, 0.054901960784313725)])
plt.xticks([-180,-90,0,90,180])
ax.set_xlim(-180, 180)
ax.text(-15,35,"****")
#ax.set_ylim(0, 15)
ax.tick_params(width=2)
ax.legend(title="",frameon=False)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.ylabel("Count")
plt.xlabel("Phase (deg)")
#plt.show()
plt.savefig('../output/phase_modulation_analysis/mean_phase_dist_sig_pyr.png', dpi=300)

"""ax.set(xlabel="", ylabel="")
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.savefig('../output/phase_modulation_analysis/mean_phase_dist_sig_pyr_notext.png', dpi=300)"""


plt.figure()
plt.rcParams['font.size'] = '14'

phase_df_sig = phase_df[(phase_df['sig'] == True) & (phase_df['class'] == "INT") & (phase_df['session'] == 'combined')]
wt = phase_df[(phase_df['strain'] == 'wt') & (phase_df['sig'] == True) & (phase_df['class'] == "INT")]
tg = phase_df[(phase_df['strain'] == 'tg') & (phase_df['sig'] == True) & (phase_df['class'] == "INT")]

#print(stats.ks_2samp(tg['mean'].to_numpy(), wt['mean'].to_numpy()))

## compare stats between two distributions




print(kuiper.kuiper_two(wt['mean'].to_numpy(), tg['mean'].to_numpy()))
ax = sns.histplot(phase_df_sig, x="mean", hue="strain", binwidth=bin_width, binrange=(bins[0],bins[-1]), linewidth=1, element="step",palette=[(0.6509803921568628, 0.807843137254902, 0.8901960784313725),(0.9921568627450981, 0.7490196078431373, 0.43529411764705883)])


ax.text(-5,14,"ns")
ax.set_ylim(0, 15)
ax.set_xlim(-180, 180)

ax.legend(title="",frameon=False)


plt.xticks([-180,-90,0,90,180])
ax.tick_params(width=2)

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.ylabel("Count")
plt.xlabel("Phase (deg)")
#plt.show()
plt.savefig('../output/phase_modulation_analysis/mean_phase_dist_sig_int.png', dpi=300)

"""ax.set(xlabel="", ylabel="")
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.savefig('../output/phase_modulation_analysis/mean_phase_dist_sig_int_notext.png', dpi=300)"""
