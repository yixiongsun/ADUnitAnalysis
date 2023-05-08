import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import unit_utils
import os

import scipy.io as sio
import seaborn as sns
from scipy import stats
from statannotations.Annotator import Annotator
import subjects

dfs = []
# read csv from

ids = ['18-1', '25-10', '36-3', '36-1', '64-30', '59-2', '62-1']

for id in ids:
    subject = subjects.subjects[id]

    for task in subjects.sessions:
        for session in subjects.sessions[task]:
            dfs.append(pd.read_csv(os.path.join(subject[task + 'dir'], session, 'firing_rates.csv')))

df = pd.concat(dfs)
df = df.reset_index()


#firing_rates_df = pd.read_csv('../output/firing_rates.csv')

sns.set_style('ticks')
plt.rcParams['font.size'] = '16'
plt.rcParams['axes.linewidth'] = 2
plt.figure(figsize=(6,8))
pyr_df = df[df['class'] == 'pyr']
pyr_df["fr"] = pyr_df['spikes'] / pyr_df['duration']
pyr_df["fr"] = pyr_df["fr"].apply(lambda x: np.log(1+x))

pyr_df = pyr_df[pyr_df['session']!='Sleep3']
pyr_df = pyr_df[pyr_df['task']=='hab']
pyr_df = pyr_df[pyr_df['Interval']=='Ripple']
#print(np.amax(pyr_firing_rates["fr"]))
#print(np.amin(pyr_firing_rates["fr"]))


ax = sns.violinplot(x='strain', y='fr', hue='session', data=pyr_df, split=True, inner=None)

h,l = ax.get_legend_handles_labels()

#ax.legend(title='Strain', loc='upper left', labels=['WT', 'TG'])
sns.pointplot(
    data=pyr_df, x='strain', y='fr', hue='session',
    errorbar='se', capsize=0.05, join=False, color="black", dodge=0.2,scale=0.75,errwidth=1.5#, legend=False
)

ax.legend(loc='upper left', handles=h,labels=['WT', 'TG'], frameon=False)
ax.set_ylim(-1, 5)
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.tick_params(width=2)

pairs = [
    (("wt", "Sleep1"), ("wt", "Sleep2")),
    (("tg", "Sleep1"), ("tg", "Sleep2")),
]
annot = Annotator(ax, pairs, data=pyr_df, x='strain', y='fr', hue='session')
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction='bonferroni',
                text_format='star', loc="outside").apply_test().annotate()



ax.set(xlabel="", ylabel='Log firing rate (Hz)')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)

plt.show()
#plt.title('Firing rates of putative pyramidal cells during and outside of ripples between TG and WT', y=1.08)
#plt.savefig(fname='../output/firing_rate_analysis/firing_rates_pyr.png',bbox_inches='tight', dpi=300)

