import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import unit_utils
import os

import scipy.io as sio
import seaborn as sns
from scipy import stats
from statannotations.Annotator import Annotator


firing_rates_df = pd.read_csv('../output/firing_rates.csv')


di_firing_rates = firing_rates_df[firing_rates_df['Interval'] == 'Ripple'].reset_index()

nrfr = firing_rates_df[firing_rates_df['Interval'] == 'Non Ripple']['fr'].reset_index()['fr']
rfr = di_firing_rates['fr']

di = rfr / nrfr

di_firing_rates['di'] = di

pyr_di_firing_rates = di_firing_rates[(di_firing_rates['class'] == 'PYR') & (di_firing_rates['session'] != 'OLM Rec 3')]

pyr_di_firing_rates["di"] = pyr_di_firing_rates["di"].apply(lambda x: np.log(1+x))


sns.set_style('ticks')

plt.rcParams['font.size'] = '16'
plt.figure(figsize=(6,8))
ax = sns.violinplot(x='strain', y='di', data=pyr_di_firing_rates)
ax.set_ylim(0, 4)
ax.set_yticks([0, 1, 2, 3, 4])
pairs = [
    ("WT", "TG"),
]
annot = Annotator(ax, pairs, data=pyr_di_firing_rates, x='strain', y='di')
annot.configure
annot.configure(test="Mann-Whitney", comparisons_correction='bonferroni',
                text_format='star', loc="outside").apply_test().annotate()

ax.set(xlabel='Recording Sessions', ylabel='Fold increase')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.grid(False)
plt.savefig(fname='../output/firing_rate_analysis/pyr_increase_firing_rates.png',bbox_inches='tight', dpi=300)

"""ax.set(xlabel="", ylabel="")
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.savefig(fname='../output/firing_rate_analysis/pyr_increase_firing_rates_notext.png',bbox_inches='tight', dpi=300)"""