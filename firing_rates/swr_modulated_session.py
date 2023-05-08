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

        task_df = []
        for session in subjects.sessions[task]:
            if session == 'Sleep3' or session == 'combined':
                continue
            task_df.append(pd.read_csv(os.path.join(subject[task + 'dir'], session, 'swr_modulation.csv')))


        df = pd.DataFrame()
        df['task'] = task_df[0]['task']
        df['unit'] = task_df[0]['unit']
        df['class'] = task_df[0]['class']
        df['strengths'] = (task_df[1]['strengths'] - task_df[0]['strengths'])
        df['strain'] = task_df[0]['strain']

        #print(df)
        dfs.append(df)



df = pd.concat(dfs)
df = df.reset_index()
#df = df.groupby(['id', 'task', 'class']).apply(lambda x: x/x.sum())





# compare baseline WT vs TG during hab 1
#df[(df['task'] == 'hab') & (df['session'] == 'Sleep1')]


# filter by increased only
# look at pyr increase in hab and olm

pyr_inc_df = df[(df['class'] == 'pyr')]
sns.set_style('ticks')
plt.rcParams['font.size'] = '16'
plt.rcParams['axes.linewidth'] = 2

plt.figure(figsize=(6,8))
ax = sns.violinplot(data=pyr_inc_df, y='strengths', x='strain', hue='task',split=True)
#(('hab', 'tg'), ('hab', 'wt')), (('olm', 'tg'), ('olm', 'wt')), (('hab', 'wt'), ('olm', 'wt')), (('hab', 'tg'), ('olm', 'tg'))
annot = Annotator(ax, [(('wt', 'hab'), ('wt', 'olm')), (('tg', 'hab'), ('tg', 'olm'))], data=pyr_inc_df, y='strengths', x='strain', hue='task')
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction='bonferroni',
                text_format='star', loc="outside").apply_test().annotate()

plt.show()


pyr_inc_df = df[(df['class'] == 'int')]
sns.set_style('ticks')
plt.rcParams['font.size'] = '16'
plt.rcParams['axes.linewidth'] = 2

plt.figure(figsize=(6,8))
ax = sns.violinplot(data=pyr_inc_df, y='strengths', x='strain', hue='task',split=True)
annot = Annotator(ax, [(('wt', 'hab'), ('wt', 'olm')), (('tg', 'hab'), ('tg', 'olm'))], data=pyr_inc_df, y='strengths', x='strain', hue='task')

#annot = Annotator(ax, [(('hab', 'tg'), ('hab', 'wt')), (('olm', 'tg'), ('olm', 'wt')), (('hab', 'wt'), ('olm', 'wt')), (('hab', 'tg'), ('olm', 'tg'))], data=pyr_inc_df, y='strengths', x='task', hue='strain')
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction='bonferroni',
                text_format='star', loc="outside").apply_test().annotate()

plt.show()






# more ripples identified in TG mice

# need to count only sleep periods -> maybe TG mice sleep more?

