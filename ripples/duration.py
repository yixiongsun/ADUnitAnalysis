import subjects
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

import numpy as np

# compare incidence rates across strain + session + plot figures

# load files for subjects and append to Dataframe
dfs = []
for id in subjects.subjects:
    subject = subjects.subjects[id]

    for task in subjects.sessions:
        for session in subjects.sessions[task]:

            dfs.append(pd.read_csv(os.path.join(subject[task + 'dir'], session, 'ripple_durations.csv')))

df = pd.concat(dfs)


print(df)


# limit to 10 mins, 20 mins, 30 mins, 1 hour
df = df[df['session'] != 'Sleep3']
df_10 = df[df['onsets'] < 10 * 60]
df_20 = df[df['onsets'] < 20 * 60]
df_30 = df[df['onsets'] < 30 * 60]
df_60 = df[df['onsets'] < 60 * 60]



# log transform durations
#df['durations'] = df.apply(lambda row: np.log(row['durations']), axis=1)

# compare baseline WT vs TG during hab 1
#df[(df['task'] == 'hab') & (df['session'] == 'Sleep1')]
#sns.violinplot(data=df[(df['task'] == 'hab') & (df['session'] == 'Sleep1')], y='durations', x='strain')
#plt.show()


# compare increase from pre to post sleep hab
plt.figure()
ax = sns.violinplot(data=df_10[df_10['task'] == 'hab'], y='durations', x='strain', hue='session')
sns.swarmplot(data=df_10[df_10['task'] == 'hab'], y='durations', x='strain', hue='session', dodge=True)
annot = Annotator(ax, [(("wt", "Sleep1"), ("wt","Sleep2")), (("tg", "Sleep1"), ("tg","Sleep2"))], data=df_10[df_10['task'] == 'hab'], y='durations', x='strain', hue='session')
annot.configure
annot.configure(test="Mann-Whitney",#custom_test, comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
plt.show()
plt.figure()
ax = sns.violinplot(data=df_10[df_10['task'] == 'olm'], y='durations', x='strain', hue='session')
sns.swarmplot(data=df_10[df_10['task'] == 'olm'], y='durations', x='strain', hue='session', dodge=True)
annot = Annotator(ax, [(("wt", "Sleep1"), ("wt","Sleep2")), (("tg", "Sleep1"), ("tg","Sleep2"))], data=df_10[df_10['task'] == 'olm'], y='durations', x='strain', hue='session')
annot.configure
annot.configure(test="Mann-Whitney",#custom_test, comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
plt.show()

plt.figure()
ax = sns.violinplot(data=df_20[df_20['task'] == 'hab'], y='durations', x='strain', hue='session')
sns.swarmplot(data=df_20[df_20['task'] == 'hab'], y='durations', x='strain', hue='session', dodge=True)
annot = Annotator(ax, [(("wt", "Sleep1"), ("wt","Sleep2")), (("tg", "Sleep1"), ("tg","Sleep2"))], data=df_20[df_20['task'] == 'hab'], y='durations', x='strain', hue='session')
annot.configure
annot.configure(test="Mann-Whitney",#custom_test, comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
plt.show()
plt.figure()
ax = sns.violinplot(data=df_20[df_20['task'] == 'olm'], y='durations', x='strain', hue='session')
sns.swarmplot(data=df_20[df_20['task'] == 'olm'], y='durations', x='strain', hue='session', dodge=True)
annot = Annotator(ax, [(("wt", "Sleep1"), ("wt","Sleep2")), (("tg", "Sleep1"), ("tg","Sleep2"))], data=df_20[df_20['task'] == 'olm'], y='durations', x='strain', hue='session')
annot.configure
annot.configure(test="Mann-Whitney",#custom_test, comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
plt.show()

plt.figure()
sns.violinplot(data=df_30[df_30['task'] == 'hab'], y='durations', x='strain', hue='session')
plt.figure()
sns.violinplot(data=df_30[df_30['task'] == 'olm'], y='durations', x='strain', hue='session')
plt.show()

plt.figure()
sns.violinplot(data=df_60[df_60['task'] == 'hab'], y='durations', x='strain', hue='session')
plt.figure()
sns.violinplot(data=df_60[df_60['task'] == 'olm'], y='durations', x='strain', hue='session')
plt.show()

#sns.violinplot(data=df_10[df_10['task'] == 'hab'], y='durations', x='strain', hue='task')
#plt.show()

# more ripples identified in TG mice

# need to count only sleep periods -> maybe TG mice sleep more?