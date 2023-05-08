import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd

import unit_utils
import os

import scipy.io as sio
from scipy import stats
import pickle

from sklearn.decomposition import FastICA
import seaborn as sns
from analysis import files
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
from scipy import stats
from scipy import signal

sns.set_style('ticks')



pre = True

if pre:

    directory = '../output/assemblies_pre_template.pkl'
else:
    directory = '../output/assemblies_post_template.pkl'

assemblies = pickle.load(open(directory, 'rb'))

reactivation_strengths = []

data = {
    'strength': [],
    'session': [],
    'strain': [],
    'z_strength': [],
    'relative_change': [],
    'corr': [],
    'dcorr': [],
    'activation_strength': []
}

for strain in assemblies:
    for assembly in assemblies[strain]:
        session = assembly['session']



        z_matrix = assembly['binned']
        templates = assembly['templates']
        num_templates = templates.shape[1]

        strengths = assembly['reactivation']
        mean_s = np.mean(strengths, axis=1)
       # print(mean_s)


        z_strengths = stats.zscore(strengths, axis=1)

        r_s = np.mean(strengths[:, assembly['ripple_bins']], axis=1)
        r_i = (r_s - mean_s)/mean_s
        r_z = np.mean(z_strengths[:, assembly['ripple_bins']], axis=1)

        activations = unit_utils.reactivation_indices(z_strengths)
        # get strengths of activations

        a_s = []
        for k in range(0, num_templates):
            a_s.append(np.mean(z_strengths[k,activations[k]]))

        _, _, max_corr, diff = unit_utils.activation_correlation(num_templates, z_strengths, activations, assembly['ripple_bins'])
        #print(diff)
        #print(max_corr)

        # plt.show()

        for k in range(0, num_templates):
            #data['strength'].append(r_s[k])
            data['strength'].append(mean_s[k])
            data['strain'].append(strain)
            data['session'].append(session)
            data['z_strength'].append(r_z[k])
            data['relative_change'].append(r_i[k])
            data['corr'].append(max_corr[k])
            data['dcorr'].append(diff[k])
            data['activation_strength'].append(a_s[k])

        # Try separating assemblies based on corr -> weak vs strong assemblies

df = pd.DataFrame(data)

# manipulate df

df1 = df[(df['session'] == "Hab Rec 1")]
df2 = df[(df['session'] == "Hab Rec 2")]
df1 = df1.reset_index(drop=True)
df2 = df2.reset_index(drop=True)

df1['strength_post'] = df2['strength']
df1['z_strength_post'] = df2['z_strength']
df1['relative_change_post'] = df2['relative_change']
df1['corr_post'] = df2['corr']
df1['dcorr_post'] = df2['dcorr']


hab_diff_df = df1
hab_diff_df['delta_strength'] = hab_diff_df.apply(lambda row: row['strength_post'] - row['strength'], axis=1)
hab_diff_df['delta_z_strength'] = hab_diff_df.apply(lambda row: row['z_strength_post'] - row['z_strength'], axis=1)
hab_diff_df['delta_relative_change'] = hab_diff_df.apply(lambda row: row['relative_change_post'] - row['relative_change'], axis=1)
hab_diff_df['delta_corr'] = hab_diff_df.apply(lambda row: row['corr_post'] - row['corr'], axis=1)
hab_diff_df['delta_dcorr'] = hab_diff_df.apply(lambda row: row['dcorr_post'] - row['dcorr'], axis=1)
hab_diff_df['activation_type'] = hab_diff_df.apply(lambda row: 1 if row['dcorr'] > 0.06 else 0, axis=1)


# filter out high vs low
#hab_diff_df = hab_diff_df[hab_diff_df['activation_type'] == 1]
"""plt.figure()
ax = sns.swarmplot(data=hab_diff_df, x="strain", y="delta_strength", dodge=True)

pairs = [
    ("WT", "TG")
]
annot = Annotator(ax, pairs, data=hab_diff_df, x="strain", y="delta_strength")
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction='bonferroni',
                text_format='star', loc="outside").apply_test().annotate()
plt.show()"""



df1 = df[(df['session'] == "OLM Rec 1")]
df2 = df[(df['session'] == "OLM Rec 2")]
df1 = df1.reset_index(drop=True)
df2 = df2.reset_index(drop=True)

df1['strength_post'] = df2['strength']
df1['z_strength_post'] = df2['z_strength']
df1['relative_change_post'] = df2['relative_change']
df1['corr_post'] = df2['corr']
df1['dcorr_post'] = df2['dcorr']


olm_diff_df = df1
olm_diff_df['delta_strength'] = olm_diff_df.apply(lambda row: row['strength_post'] - row['strength'], axis=1)
olm_diff_df['delta_z_strength'] = olm_diff_df.apply(lambda row: row['z_strength_post'] - row['z_strength'], axis=1)
olm_diff_df['delta_relative_change'] = olm_diff_df.apply(lambda row: row['relative_change_post'] - row['relative_change'], axis=1)
olm_diff_df['delta_corr'] = olm_diff_df.apply(lambda row: row['corr_post'] - row['corr'], axis=1)
olm_diff_df['delta_dcorr'] = olm_diff_df.apply(lambda row: row['dcorr_post'] - row['dcorr'], axis=1)
olm_diff_df['activation_type'] = olm_diff_df.apply(lambda row: 1 if row['dcorr'] > 0.06 else 0, axis=1)

#olm_diff_df = olm_diff_df[olm_diff_df['activation_type'] == 1]

df = pd.concat([hab_diff_df, olm_diff_df], ignore_index=True)

wtdf = df[df['strain'] == 'WT']


plt.figure()
ax = sns.violinplot(data=wtdf, x="session", y="delta_relative_change", dodge=True, palette='Blues')

pairs = [
    ("Hab Rec 1", "OLM Rec 1")
]
annot = Annotator(ax, pairs, data=wtdf, x="session", y="delta_relative_change")
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction='bonferroni',
                text_format='star', loc="outside").apply_test().annotate()
plt.savefig('../output/assembly/wt_experience_delta_relative_change.png', dpi=300)


tgdf = df[df['strain'] == 'TG']


plt.figure()
ax = sns.violinplot(data=tgdf, x="session", y="delta_relative_change", dodge=True, palette='Oranges')

pairs = [
    ("Hab Rec 1", "OLM Rec 1")
]
annot = Annotator(ax, pairs, data=tgdf, x="session", y="delta_relative_change")
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction='bonferroni',
                text_format='star', loc="outside").apply_test().annotate()
plt.savefig('../output/assembly/tg_experience_delta_relative_change.png', dpi=300)
