import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest

from scipy import stats

custom_long_name = 'KS 2 samp statistical test '
custom_short_name = 'KS'
custom_func = stats.ks_2samp
custom_test = StatTest(custom_func, custom_long_name, custom_short_name)

phase_df = pd.read_csv('../output/phase_info.csv')

#phase_df = phase_df.replace({"Hab Rec 1_1": "Hab Rec 1","Hab Rec 2_1": "Hab Rec 2","OLM Rec 1_1": "OLM Rec 1","OLM Rec 2_1": "OLM Rec 2","OLM Rec 3_1": "OLM Rec 3"})


phase_df_sig = phase_df[phase_df['sig'] == True]


# repeat by separating int and pyr
#plt.figure(figsize=(6,8))

# select hab 1 and olm 1
df = phase_df_sig[(phase_df_sig['session'] == 'combined')]
#df['Condition'] = df.apply(lambda row: row['strain'] + " " + row['class'], axis=1)
df['z_value'] = df.apply(lambda row: np.log(row['z_value']), axis=1)
df = df[df['num_spikes'] > 50]
plt.rcParams['font.size'] = '16'
plt.rcParams['axes.linewidth'] = 2

"""ax = sns.ecdfplot(df, x="z_value", hue="Condition")
plt.xlabel("Strength of phase locking (log z value)")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.tight_layout
plt.savefig('../output/phase_modulation_analysis/z_cdf_pyr.png', dpi=300)"""

"""ax.set(xlabel="", ylabel="")
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.savefig('../output/phase_modulation_analysis/z_cdf_pyr_notext.png', dpi=300)"""

plt.figure(figsize=(4,8))
ax = sns.violinplot(data=df, x="class", y='z_value', hue="strain", legend=False, split=True, inner=None, palette=[(0.6509803921568628, 0.807843137254902, 0.8901960784313725),(0.9921568627450981, 0.7490196078431373, 0.43529411764705883)])

h,l = ax.get_legend_handles_labels()

sns.pointplot(
    data=df, x="class", y='z_value', hue="strain",
    errorbar='se', capsize=0.05, join=False, color="black", dodge=0.2,scale=0.75,errwidth=1.5
)
#ax.set_ylim(0, 10)
ax.tick_params(width=2)

annot = Annotator(ax, [(("PYR", "wt"), ("PYR","tg")), (("INT", "wt"), ("INT","tg"))], data=df, x="class", y='z_value', hue="strain")
annot.configure
annot.configure(test="Mann-Whitney",#custom_test, comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()


ax.legend(loc='upper left', handles=h,labels=['WT', 'TG'], frameon=False)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
ax.set_xticks([0,1], ['PYR', 'INT'])
plt.xlabel("")
plt.ylabel("Phase locking strength")
plt.tight_layout()
#plt.show()
plt.savefig('../output/phase_modulation_analysis/ripple_phase_strength/z_strength.png', dpi=300)


print(len(df[(df['class'] == 'PYR') & (df['strain'] == 'wt')]))
print(len(df[(df['class'] == 'INT') & (df['strain'] == 'wt')]))
print(len(df[(df['class'] == 'PYR') & (df['strain'] == 'tg')]))
print(len(df[(df['class'] == 'INT') & (df['strain'] == 'tg')]))


"""ax.set(xlabel="", ylabel="")
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.savefig('../output/phase_modulation_analysis/z_violin_pyr_notext.png', dpi=300)
"""

"""plt.figure(figsize=(4,8))
df = phase_df_sig[((phase_df_sig['session'] == 'OLM Rec 2') | (phase_df_sig['session'] == 'Hab Rec 2')) & (phase_df_sig['class'] == 'INT')]
df['Condition'] = df.apply(lambda row: row['strain'] + " " + row['class'], axis=1)
df['z_value'] = df.apply(lambda row: np.log(row['z_value']), axis=1)
df = df[df['num_spikes'] > 100]

plt.rcParams['font.size'] = '16'"""
"""ax = sns.ecdfplot(df, x="z_value", hue="Condition")
plt.xlabel("Strength of phase locking (log z value)")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.tight_layout
plt.savefig('../output/phase_modulation_analysis/z_cdf_int.png', dpi=300)"""

"""ax.set(xlabel="", ylabel="")
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.savefig('../output/phase_modulation_analysis/z_cdf_int_notext.png', dpi=300)"""

"""plt.figure(figsize=(4,8))
ax = sns.violinplot(data=df, x="strain", y="ppc", legend=False)
#ax.set_ylim(0, 10)
ax.tick_params(width=2)

annot = Annotator(ax, [("WT", "TG")], data=df, x='strain', y='ppc')
annot.configure
annot.configure(test=custom_test, comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.xlabel("Strain")
plt.ylabel("Strength of phase locking (log z value)")
plt.savefig('../output/phase_modulation_analysis/ripple_phase_strength/z_violin_int.png', dpi=300)"""
"""
ax.set(xlabel="", ylabel="")
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.savefig('../output/phase_modulation_analysis/z_violin_int_notext.png', dpi=300)"""