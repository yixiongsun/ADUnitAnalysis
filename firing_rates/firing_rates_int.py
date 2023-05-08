import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import unit_utils
import os
import subjects
import scipy.io as sio
import seaborn as sns
from scipy import stats
from statannotations.Annotator import Annotator
from subjects import subjects


dfs = []
# read csv from
for subject in ['18-1', '25-10', '36-3', '36-1', '64-30', '59-2', '62-1']:
    # directory

    # combine hab sleep 1 and sleep 2, and olm sleep 1 and sleep 2
    habs1 = pd.read_csv(os.path.join(subjects[subject]['habdir'], 'Sleep1', 'firing_rates.csv'))
    habs2 = pd.read_csv(os.path.join(subjects[subject]['habdir'], 'Sleep2', 'firing_rates.csv'))

    # combine based on sum
    habdf = pd.concat([habs1, habs1]).groupby(['unit', 'id', 'Interval', 'class', 'strain']).sum().reset_index()
    habdf['fr'] = habdf['spikes']/habdf['duration']

    # combine hab sleep 1 and sleep 2, and olm sleep 1 and sleep 2
    olms1 = pd.read_csv(os.path.join(subjects[subject]['olmdir'], 'Sleep1', 'firing_rates.csv'))
    olms2 = pd.read_csv(os.path.join(subjects[subject]['olmdir'], 'Sleep2', 'firing_rates.csv'))

    # combine based on sum
    olmdf = pd.concat([olms1, olms2]).groupby(['unit', 'id', 'Interval', 'class', 'strain']).sum().reset_index()
    olmdf['fr'] = olmdf['spikes'] / olmdf['duration']
    dfs.append(habdf)
    dfs.append(olmdf)


df = pd.concat(dfs)
#firing_rates_df = pd.read_csv('../output/firing_rates.csv')


sns.set_style('ticks')
plt.rcParams['font.size'] = '16'
plt.rcParams['axes.linewidth'] = 2

plt.figure(figsize=(6,8))
int_df = df[df['class'] == 'int']
int_df["fr"] = int_df["fr"].apply(lambda x: np.log(1+x))

print(sns.color_palette("Paired"))

ax = sns.violinplot(x='Interval', y='fr', hue='strain', data=int_df, split=True, inner=None,palette=[(0.6509803921568628, 0.807843137254902, 0.8901960784313725),(0.9921568627450981, 0.7490196078431373, 0.43529411764705883)])
#plt.setp(ax.collections, alpha=.7)
h,l = ax.get_legend_handles_labels()

#ax.legend(title='Strain', loc='upper left', labels=['WT', 'TG'])
sns.pointplot(
    data=int_df, x='Interval', y='fr', hue='strain',
    errorbar='se', capsize=0.05, join=False, color="black", dodge=0.2,scale=0.75,errwidth=1.5#, legend=False
)
pairs = [
    (("Ripple", "wt"), ("Ripple", "tg")),
    (("Non Ripple", "wt"), ("Non Ripple", "tg")),
]
annot = Annotator(ax, pairs, data=int_df, x='Interval', y='fr', hue='strain')
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction='bonferroni',
                text_format='star', loc="outside").apply_test().annotate()

# print # of each
print("Ripple wt" + str(((int_df['Interval'] == 'Ripple') & (int_df['strain'] == 'wt')).sum()))
print("Ripple tg" + str(((int_df['Interval'] == 'Ripple') & (int_df['strain'] == 'tg')).sum()))
print("Non Ripple wt" + str(((int_df['Interval'] == 'Non Ripple') & (int_df['strain'] == 'wt')).sum()))
print("Non Ripple tg" + str(((int_df['Interval'] == 'Non Ripple') & (int_df['strain'] == 'tg')).sum()))


ax.legend(loc='upper left', handles=h,labels=['WT', 'TG'], frameon=False)
ax.set(xlabel="", ylabel='Log firing rate (Hz)')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
ax.set_ylim(-1, 5)
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.tick_params(width=2)

#plt.show()

#plt.title('Firing rates of putative interneurons during and outside of ripples between TG and WT', y=1.08)
plt.savefig(fname='../output/firing_rate_analysis/firing_rates_int.png',bbox_inches='tight', dpi=300)


"""ax.set(xlabel="", ylabel="")
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.savefig(fname='../output/firing_rate_analysis/firing_rates_int_notext.png',bbox_inches='tight', dpi=300)"""
