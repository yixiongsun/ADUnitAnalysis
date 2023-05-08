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


assemblies = []
new = False
if os.path.exists('../output/binned_20ms.pkl') and new == False:
    assemblies = pickle.load(open('../output/binned_20ms.pkl', 'rb'))
else:
    data = {}
    for strain in files:
        z = []
        for k in files[strain]:
            rating_file = k['rating_file']
            directories = k['directories']
            tetrodes = k['tetrodes']
            sessions = k['sessions']


            binsize = 20




            for d in range(0,len(directories)):
                print(strain + " " + sessions[d])

                directory = directories[d]


                units = unit_utils.good_units(os.path.dirname(directory), rating_file, tetrodes)


                # Z-scored bins
                # try bin size of 100
                z_hab_s1, spike_matrix, bins = unit_utils.z_spike_matrix(directory, units, 100, remove_nan=True)
                #



                break



        data[strain] = z

    pickle.dump(data, open('../output/binned_20ms.pkl', 'wb'))