# plot histogram of good neuron in 18-1
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import unit_utils
import subjects


bins = np.linspace(-np.pi, np.pi, 33)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_widths = bins[1:] - bins[:-1]


data = subjects.load_data(subjects.subjects['18-1'], units=True, phases=True)

pyr_units = data['olm']['int_units']



ripple_phases = data['olm']['Sleep2']['ripple_phases']
ripple_windows = data['olm']['Sleep2']['ripple_windows']
ts = data['olm']['Sleep2']['ts']

pyr_spike_phases = unit_utils.spike_phases(ripple_phases, ripple_windows,ts,pyr_units)

# select unit # 10


for s in pyr_spike_phases:

    sns.histplot(data=s, binwidth=bin_widths[0], binrange=(bins[0],bins[-1]), linewidth=0.5)
    plt.show()


# fig num 1, 3