import subjects
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# compare incidence rates across strain + session + plot figures

# load files for subjects and append to Dataframe
dfs = []
for id in subjects.subjects:
    subject = subjects.subjects[id]
    id_df = []
    for task in subjects.sessions:
        for session in subjects.sessions[task]:

            id_df.append(pd.read_csv(os.path.join(subject[task + 'dir'], session, 'ripple_incidence.csv')))

    df = pd.concat(id_df).groupby(['id', 'strain']).sum().reset_index()
    df['incidence'] = df['num_ripples']/df['total_time']
    dfs.append(df)

df = pd.concat(dfs)


print(df)


# compare baseline WT vs TG during hab 1
#df[(df['task'] == 'hab') & (df['session'] == 'Sleep1')]
sns.barplot(data=df, y='density', x='strain', errorbar='se')
plt.show()
print(stats.mannwhitneyu(df[df['strain'] == 'wt']['incidence'], df[df['strain'] == 'tg']['incidence']))

# compare increase from pre to post sleep hab
"""sns.barplot(data=df[(df['task'] == 'hab') & (df['session'] != 'Sleep3')], y='density_60', x='strain', hue='session')
plt.show()

sns.barplot(data=df[(df['task'] == 'olm') & (df['session'] != 'Sleep3')], y='density_60', x='strain', hue='session')
plt.show()

sns.barplot(data=df[(df['session'] == 'Sleep2')], y='density', x='strain', hue='task')
plt.show()"""

# more ripples identified in TG mice

# need to count only sleep periods -> maybe TG mice sleep more?