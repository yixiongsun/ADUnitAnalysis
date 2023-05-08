import subjects
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statannotations.Annotator import Annotator

"""
To compute SWR modulation of CA1 neurons, we aligned spikes to the onset of SWR and averaged the firing rates across all events in 10 ms bins in a 1 s window centered around SWR onset. The binned firing rates were z-scored based on the average rates during the first 200 ms. Neurons were categorized as SWR-excited or SWR-inhibited if its firing rate in a 200-ms window from SWR onset surpassed a positive or negative threshold of 2 standard deviations. Neurons that did not surpass the threshold in either direction were considered to be not modulated by SWRs (non-modulated). 
"""


# compare incidence rates across strain + session + plot figures

# load files for subjects and append to Dataframe
ids = ['18-1', '25-10', '36-3', '36-1', '64-30', '59-2', '62-1']

dfs = []
for id in ids:
    subject = subjects.subjects[id]

    for task in subjects.sessions:

        dfs.append(pd.read_csv(os.path.join(subject[task + 'dir'], 'swr_modulation.csv')))

df = pd.concat(dfs)
df = df.reset_index()

print(df)

def modulation_counts(df):
    inc = df['modulation'] == 'increased'
    inc = inc.values.sum()
    dec = df['modulation'] == 'decreased'
    dec = dec.values.sum()

    total = len(df['modulation'])

    return inc, dec, total


# count number of increased, decreased for each strain and class

fig, axs = plt.subplots(2, 2)


wt_int = df[(df['strain'] == 'wt') & (df['class'] == 'int')]

inc, dec, total = modulation_counts(wt_int)

labels = ['increased', 'decreased', 'none']
sizes = [inc, dec, total - inc - dec]
axs[0,0].pie(sizes, labels=labels)


wt_pyr = df[(df['strain'] == 'wt') & (df['class'] == 'pyr')]

inc, dec, total = modulation_counts(wt_pyr)
labels = ['increased', 'decreased', 'none']
sizes = [inc, dec, total - inc - dec]
axs[0,1].pie(sizes, labels=labels)
tg_int = df[(df['strain'] == 'tg') & (df['class'] == 'int')]

inc, dec, total = modulation_counts(tg_int)
labels = ['increased', 'decreased', 'none']
sizes = [inc, dec, total - inc - dec]
axs[1,0].pie(sizes, labels=labels)

tg_pyr = df[(df['strain'] == 'tg') & (df['class'] == 'pyr')]

inc, dec, total = modulation_counts(tg_pyr)
labels = ['increased', 'decreased', 'none']
sizes = [inc, dec, total - inc - dec]
axs[1,1].pie(sizes, labels=labels)

#plt.show()

# compare baseline WT vs TG during hab 1
#df[(df['task'] == 'hab') & (df['session'] == 'Sleep1')]


# filter by increased and decreased only
inc_df = df#[(df['modulation'] == 'increased')]
sns.set_style('ticks')
plt.rcParams['font.size'] = '16'
plt.rcParams['axes.linewidth'] = 2

plt.figure(figsize=(6,8))
ax = sns.violinplot(data=inc_df, y='strengths', x='class', hue='strain', split=True, inner=None, palette=[(0.6509803921568628, 0.807843137254902, 0.8901960784313725),(0.9921568627450981, 0.7490196078431373, 0.43529411764705883)])

h,l = ax.get_legend_handles_labels()

#ax.legend(title='Strain', loc='upper left', labels=['WT', 'TG'])
sns.pointplot(
    data=inc_df, y='strengths', x='class', hue='strain',
    errorbar='se', capsize=0.05, join=False, color="black", dodge=0.2,scale=0.75,errwidth=1.5#, legend=False
)
ax.legend(loc='upper left', handles=h,labels=['WT', 'TG'], frameon=False)
#ax.set_ylim(-1, 5)
#ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.tick_params(width=2)
ax.set_xticks([0, 1], ['PYR', 'INT'])
ax.set(xlabel="", ylabel='Modulation strength')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)

annot = Annotator(ax, [(('pyr', 'wt'), ('pyr','tg')), (('int', 'wt'), ('int','tg'))], data=inc_df, y='strengths', x='class', hue='strain')
annot.configure(test="Mann-Whitney", comparisons_correction='bonferroni',
                text_format='star', loc="outside").apply_test().annotate()
plt.tight_layout()
#plt.show()
plt.savefig(fname='../../output/firing_rate_analysis/modulated.png',bbox_inches='tight', dpi=300)







# more ripples identified in TG mice

# need to count only sleep periods -> maybe TG mice sleep more?

