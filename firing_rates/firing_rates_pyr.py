import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import unit_utils
import os

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
pyr_df = df[df['class'] == 'pyr']
pyr_df["fr"] = pyr_df["fr"].apply(lambda x: np.log(1+x))

#print(np.amax(pyr_firing_rates["fr"]))
#print(np.amin(pyr_firing_rates["fr"]))
print(sns.color_palette())


ax = sns.violinplot(x='Interval', y='fr', hue='strain', data=pyr_df, split=True, inner=None, palette=[(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (1.0, 0.4980392156862745, 0.054901960784313725)])

h,l = ax.get_legend_handles_labels()

#ax.legend(title='Strain', loc='upper left', labels=['WT', 'TG'])
sns.pointplot(
    data=pyr_df, x='Interval', y='fr', hue='strain',
    errorbar='se', capsize=0.05, join=False, color="black", dodge=0.2,scale=0.75,errwidth=1.5#, legend=False
)

ax.legend(loc='upper left', handles=h,labels=['WT', 'TG'], frameon=False)
ax.set_ylim(-1, 5)
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.tick_params(width=2)

pairs = [
    (("Ripple", "wt"), ("Ripple", "tg")),
    (("Non Ripple", "wt"), ("Non Ripple", "tg")),
]
annot = Annotator(ax, pairs, data=pyr_df, x='Interval', y='fr', hue='strain')
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction='bonferroni',
                text_format='star', loc="outside").apply_test().annotate()

print("Ripple wt" + str(((pyr_df['Interval'] == 'Ripple') & (pyr_df['strain'] == 'wt')).sum()))
print("Ripple tg" + str(((pyr_df['Interval'] == 'Ripple') & (pyr_df['strain'] == 'tg')).sum()))
print("Non Ripple wt" + str(((pyr_df['Interval'] == 'Non Ripple') & (pyr_df['strain'] == 'wt')).sum()))
print("Non Ripple tg" + str(((pyr_df['Interval'] == 'Non Ripple') & (pyr_df['strain'] == 'tg')).sum()))

ax.set(xlabel="", ylabel='Log firing rate (Hz)')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)

#plt.title('Firing rates of putative pyramidal cells during and outside of ripples between TG and WT', y=1.08)
plt.savefig(fname='../output/firing_rate_analysis/firing_rates_pyr.png',bbox_inches='tight', dpi=300)


"""ax.set(xlabel="", ylabel="")
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.savefig(fname='../output/firing_rate_analysis/firing_rates_pyr_notext.png',bbox_inches='tight', dpi=300)"""
