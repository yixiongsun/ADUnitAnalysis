import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#sns.set_style('ticks')
def to_ms(time_string):
    split = time_string.split(":")
    return int(split[0]) * 60 * 1000 + int(split[1]) * 1000 + int(split[2])

data = {
    "6-2": {
        "coupled": 'D:\\Behavioural\\Processed\\te1_r2_6-2.csv',
        "delayed": 'D:\\Behavioural\\Processed\\te2_r4_6-2.csv',
    },
    "8-10": {
        "coupled": 'D:\\Behavioural\\Processed\\te1_r2_8-10.csv',
        "delayed": 'D:\\Behavioural\\Processed\\te2_r4_8-10.csv'
    },
    "31-20": {
        "coupled": 'D:\\Behavioural\\Processed\\te2_r4_31-20.csv',
        "delayed": 'D:\\Behavioural\\Processed\\te1_r2_31-20.csv'
    },
    "30-22": {
        "coupled": 'D:\\Behavioural\\Processed\\te2_r4_30-22.csv',
        "delayed": 'D:\\Behavioural\\Processed\\te1_r2_30-22.csv'
    },
    "38-1": {
        "coupled": 'D:\\Behavioural\\Processed\\te1_r2_38-1.csv',
        "delayed": 'D:\\Behavioural\\Processed\\te2_r4_38-1.csv',
    }
}


DIs = []
Condition = []
ids = []
corr = []
corr_diff = []
base_corr = []

corr_df = pd.read_csv("../rippleanalysis-main/stim_corr.csv")
print(corr_df)

for mouse in data:
    mouse_corr = corr_df[corr_df['id'] == mouse]

    for condition in data[mouse]:
        df = pd.read_csv(data[mouse][condition])
        df['start'] = df['start'].map(to_ms)
        df['stop'] = df['stop'].map(to_ms)
        df['diff'] = df['stop'] - df['start']
        if condition == "delayed" and mouse == "38-1":
            new = df.loc[(df['location'] == 'TL')]['diff'].sum()
            old = df.loc[(df['location'] == 'BR')]['diff'].sum()
        else:
            new = df.loc[(df['location'] == 'BL') | (df['location'] == 'BR')]['diff'].sum()
            old = df.loc[(df['location'] == 'TR') | (df['location'] == 'TL')]['diff'].sum()
        DI = (new - old) / (new + old)
        DIs.append(DI)
        Condition.append(condition)
        ids.append(mouse)

    """corr.append(mouse_corr["stim_max"].values[0])
    corr.append(mouse_corr["delayed_stim_max"].values[0])

    corr_diff.append((mouse_corr["stim_max"] - mouse_corr["no_stim_max"]).values[0])
    corr_diff.append((mouse_corr["delayed_stim_max"] - mouse_corr["no_stim_max"]).values[0])

    base_corr.append(mouse_corr["no_stim_max"].values[0])
    base_corr.append(mouse_corr["no_stim_max"].values[0])"""


# include 2-3 month data
"""data = {
    "36-1": 'D:\\Behavioural\\Processed\\te1_r2_36-1.csv',
    "36-3": 'D:\\Behavioural\\Processed\\te1_r2_36-3.csv',
}

corr_df = pd.read_csv("../rippleanalysis-main/olm_corr.csv")

for mouse in data:
    mouse_corr = corr_df[corr_df['id'] == mouse]

    df = pd.read_csv(data[mouse])
    df['start'] = df['start'].map(to_ms)
    df['stop'] = df['stop'].map(to_ms)
    df['diff'] = df['stop'] - df['start']
    new = df.loc[(df['location'] == 'BL') | (df['location'] == 'BR')]['diff'].sum()
    old = df.loc[(df['location'] == 'TR') | (df['location'] == 'TL')]['diff'].sum()
    DI = (new - old) / (new + old)
    DIs.append(DI)
    Condition.append("3 month")
    ids.append(mouse)

    corr.append(mouse_corr["sleep2_max"].values[0])

    corr_diff.append((mouse_corr["sleep2_max"] - mouse_corr["sleep1_max"]).values[0])"""



#df = pandas.DataFrame({'Discrimination Index': DIs, 'Condition': Condition, 'IDs': ids, "corr": corr, "corr_diff": corr_diff, "base_corr": base_corr})
#print(df)

#sns.scatterplot(data=df, x="corr", y="Discrimination Index", hue="Condition")
#plt.show()

df = pandas.DataFrame({'Discrimination Index': DIs, "ids": ids,'Condition': Condition})


print(df)
df = df.sort_values(by=['Condition'],ascending=False)
plt.figure(figsize=(5,8))
plt.rcParams['font.size'] = '16'
ax = sns.stripplot(y='Discrimination Index', x='Condition', data=df,jitter=False)
sns.pointplot(y='Discrimination Index', x='Condition', data=df, join=False, scale=1, zorder=100, color='black',capsize=0.1,errorbar="ci")

locs1 = ax.get_children()[0].get_offsets()
locs2 = ax.get_children()[1].get_offsets()

for i in range(0,5):
    plt.plot([locs1[i,1],locs2[i,1]], color='grey', linewidth=0.5, linestyle='-')
    plt.plot([locs1[i,1],locs2[i,1]], color='grey', linewidth=0.5, linestyle='-')


ax.set(ylim=(-1.0, 1.0), xlim=(-0.5,1.5))
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.set_xlabel("")
ax.set_ylabel("Discrimination Index")
plt.yticks(np.arange(-1.0, 1.1, 0.2))
plt.tight_layout()
plt.show()
#plt.savefig('output/stim_olm.png', dpi=300)