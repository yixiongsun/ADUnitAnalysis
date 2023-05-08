import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import subjects
import os

# norm with log (1+x)

# TG only
all_pyr_spikes = []
all_int_spikes = []

for id in subjects.subjects:
    # directory
    subject = subjects.subjects[id]

    if subject['strain'] == 'tg':
        (pyr_spikes, int_spikes) = pickle.load(open(os.path.join(subject['habdir'], 'binned_fr.pkl'), 'rb'))
        all_pyr_spikes.append(pyr_spikes)
        all_int_spikes.append(int_spikes)
        (pyr_spikes, int_spikes) = pickle.load(open(os.path.join(subject['olmdir'], 'binned_fr.pkl'), 'rb'))
        all_pyr_spikes.append(pyr_spikes)
        all_int_spikes.append(int_spikes)







f, axes = plt.subplots(nrows=2, ncols=2, sharex='col', gridspec_kw={'height_ratios': [3,1],
                                                                         'width_ratios': [20, 1],
                                                                         'wspace': 0.1})
f.set_figheight(8)
f.set_figwidth(6)
plt.rcParams['font.size'] = '12'
z = np.concatenate(all_pyr_spikes, axis=0)
r_bin = int(z.shape[1] / 2)
print(z.shape)

z = np.log(1 + z);

max_binned = np.max(z[:,r_bin:r_bin+5], 1)



sorted_ind = np.argsort(max_binned)[::-1]
z = z[sorted_ind]

g1 = sns.heatmap(z, yticklabels=20, vmax=3, ax=axes[0][0], cbar_ax=axes[0][1], cbar_kws=dict(ticks=[0,1,2,3]))
axes[0][0].vlines(r_bin,*g1.get_ylim(),colors='white', linestyles='dashed')
g1.set_ylabel("PYR")
axes[0][0].tick_params(bottom=False)
axes[0][0].set_yticks([0.5,z.shape[0] - 0.5], [1, z.shape[0]])




z = np.concatenate(all_int_spikes, axis=0)
print(z.shape)

z = np.log(1 + z);
print(np.amax(z))
max_binned = np.max(z[:,r_bin:r_bin+5], 1)



sorted_ind = np.argsort(max_binned)[::-1]
z = z[sorted_ind]

g2 = sns.heatmap(z, yticklabels=20, vmax=4, ax=axes[1][0], cbar_ax=axes[1][1], cbar_kws=dict(ticks=[0,2,4]))
axes[1][0].vlines(r_bin,*g2.get_ylim(),colors='white', linestyles='dashed')
g2.set_ylabel("INT")
g2.set_xlabel("Time from ripple onset (s)")
axes[1][0].set_xticks([25, 50, 75], ["-.25","0",".25"], rotation=0)
axes[1][0].set_yticks([0.5,z.shape[0] - 0.5], [1, z.shape[0]])
#f.text(0.02, 0.5, 'Neuron #', ha='center', va='center', rotation='vertical')
#plt.show()
plt.savefig('../output/firing_dynamics/ripple_binned_fr_tg.png', dpi=300)

"""# remove all text
del f.texts[0]
g2.set_xlabel("")
g1.set_ylabel("")
g2.set_ylabel("")
axes[0][0].set_yticklabels([])
axes[1][0].set_yticklabels([])
axes[1][0].set_xticklabels([])
axes[0][1].set_yticklabels([])
axes[1][1].set_yticklabels([])


plt.savefig('../output/firing_dynamics/ripple_binned_fr_tg_notext.png', dpi=300)
"""
