import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
import scipy.stats as stats

custom_long_name = 'KS 2 samp statistical test '
custom_short_name = 'KS'
custom_func = stats.ks_2samp
custom_test = StatTest(custom_func, custom_long_name, custom_short_name)
plt.rcParams['font.size'] = '16'


phase_df = pd.read_csv('../output/phase_info.csv')



plt.rcParams['axes.linewidth'] = 2

#df = phase_df



df2 = phase_df[(phase_df['session'] == "Sleep2")]
df1 = phase_df[(phase_df['session'] == "Sleep1")]

df1.reset_index(drop=True)
df2.reset_index(drop=True)

df = pd.concat([df1, df2], ignore_index=True)
df_sig = pd.concat([df2["sig"] ,df1["sig"]], ignore_index=True)
df['sig_r'] = df_sig
df['sig'] = df.apply(lambda row: row['sig'] & row['sig_r'], axis=1)


df = df[(df['sig'] == True)]
df['Condition'] = df.apply(lambda row: row['strain'] + " " + row['session'], axis=1)
df["z_value"] = df.apply(lambda row: row["z_value"], axis=1)
df["z_value"] = df.apply(lambda row: row["z_value"], axis=1)
pyr_df = df[df['class'] == 'PYR']



# WT x PYR
plt.figure(figsize=(4,8))
wt_pyr_df = pyr_df[pyr_df["strain"] == "WT"]
wt_pyr_df = wt_pyr_df.reset_index(drop=True)



ax = sns.violinplot(data=wt_pyr_df, x="session", y="z_value",palette="Blues")
#ax.set_ylim(0, 10)
ax.tick_params(width=2)

annot = Annotator(ax, [("OLM Rec 1", "OLM Rec 2")], data=wt_pyr_df, x='session', y='z_value')
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
#sns.violinplot(data=wt, x="session", y="z_value")
plt.ylabel("Strength of phase locking (log z value)")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.xlabel("")

plt.tight_layout()
plt.savefig('../output/phase_modulation_analysis/OLM_violin_wt_pyr.png', dpi=300)

"""ax.set(xlabel="", ylabel="")
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.savefig('../output/phase_modulation_analysis/OLM_violin_wt_pyr_notext.png', dpi=300)"""











# TG x PYR
plt.figure(figsize=(4,8))
tg_pyr_df = pyr_df[pyr_df["strain"] == "TG"]



ax = sns.violinplot(data=tg_pyr_df, x="session", y="z_value",palette="Oranges")
#ax.set_ylim(0, 10)
ax.tick_params(width=2)

annot = Annotator(ax, [("OLM Rec 1", "OLM Rec 2")], data=tg_pyr_df, x='session', y='z_value')
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
#sns.violinplot(data=wt, x="session", y="z_value")
plt.ylabel("Strength of phase locking (log z value)")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.xlabel("")

plt.tight_layout()
plt.savefig('../output/phase_modulation_analysis/OLM_violin_tg_pyr.png', dpi=300)

"""ax.set(xlabel="", ylabel="")
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.savefig('../output/phase_modulation_analysis/OLM_violin_tg_pyr_notext.png', dpi=300)"""

int_df = df[df['class'] == 'INT']

"""plt.figure(figsize=(5,8))
ax = sns.ecdfplot(int_df, x="z_value", hue="Condition")
#sns.violinplot(data=wt, x="session", y="z_value")
plt.xlabel("Strength of phase locking (log z value)")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.tight_layout()

#plt.show()
plt.savefig('../output/phase_modulation_analysis/OLM_z_cdf_int.png', dpi=300)

plt.figure(figsize=(5,8))
ax = sns.violinplot(data=int_df,  x="Condition", y="z_value")
annot = Annotator(ax, [("WT OLM Rec 1", "WT OLM Rec 2"), ("TG OLM Rec 1", "TG OLM Rec 2")], data=pyr_df, x='Condition', y='z_value')
annot.configure
annot.configure(test=custom_test, comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
#sns.violinplot(data=wt, x="session", y="z_value")
plt.xlabel("Strength of phase locking (log z value)")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.tight_layout()

#plt.show()
plt.savefig('../output/phase_modulation_analysis/OLM_violin_cdf_int.png', dpi=300)"""


# WT x INT
plt.figure(figsize=(4,8))
wt_int_df = int_df[int_df["strain"] == "WT"]
ax = sns.violinplot(data=wt_int_df, x="session", y="z_value", palette="Blues")
#ax.set_ylim(0, 10)
ax.tick_params(width=2)

annot = Annotator(ax, [("OLM Rec 1", "OLM Rec 2")], data=wt_int_df, x='session', y='z_value')
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
#sns.violinplot(data=wt, x="session", y="z_value")
plt.ylabel("Strength of phase locking (log z value)")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.xlabel("")

plt.tight_layout()
plt.savefig('../output/phase_modulation_analysis/OLM_violin_wt_int.png', dpi=300)

"""ax.set(xlabel="", ylabel="")
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.savefig('../output/phase_modulation_analysis/OLM_violin_wt_int_notext.png', dpi=300)
"""
# TG x INT
plt.figure(figsize=(4,8))
tg_int_df = int_df[int_df["strain"] == "TG"]
ax = sns.violinplot(data=tg_int_df, x="session", y="z_value", palette="Oranges")
#ax.set_ylim(0, 10)
ax.tick_params(width=2)

annot = Annotator(ax, [("OLM Rec 1", "OLM Rec 2")], data=tg_int_df, x='session', y='z_value')
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction=None,
                text_format='star', loc="outside").apply_test().annotate()
#sns.violinplot(data=wt, x="session", y="z_value")
plt.ylabel("Strength of phase locking (log z value)")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.xlabel("")

plt.tight_layout()
plt.savefig('../output/phase_modulation_analysis/OLM_violin_tg_int.png', dpi=300)

"""ax.set(xlabel="", ylabel="")
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.savefig('../output/phase_modulation_analysis/OLM_violin_tg_int_notext.png', dpi=300)"""

pre = wt_pyr_df[wt_pyr_df["session"] == "OLM Rec 1"]["z_value"].to_numpy()
post = wt_pyr_df[wt_pyr_df["session"] == "OLM Rec 2"]["z_value"].to_numpy()
print(stats.wilcoxon(pre, post))

pre = tg_pyr_df[tg_pyr_df["session"] == "OLM Rec 1"]["z_value"].to_numpy()
post = tg_pyr_df[tg_pyr_df["session"] == "OLM Rec 2"]["z_value"].to_numpy()
print(stats.wilcoxon(pre, post))

pre = wt_int_df[wt_int_df["session"] == "OLM Rec 1"]["z_value"].to_numpy()
post = wt_int_df[wt_int_df["session"] == "OLM Rec 2"]["z_value"].to_numpy()
print(stats.wilcoxon(pre, post))

pre = tg_int_df[tg_int_df["session"] == "OLM Rec 1"]["z_value"].to_numpy()
post = tg_int_df[tg_int_df["session"] == "OLM Rec 2"]["z_value"].to_numpy()
print(stats.wilcoxon(pre, post))