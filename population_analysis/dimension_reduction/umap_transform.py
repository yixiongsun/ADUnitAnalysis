# file to run hierarchical clustering for one subject example

import matplotlib.pyplot as plt


import unit_utils

import subjects
import umap
import umap.plot

n_neighbors = 100

subject = subjects.subjects["18-1"]

data = subjects.load_data(subject, units=True)

pyr_units = data['olm']['pyr_units']

binsize = 100
ts = data['olm']['Sleep1']['ts']

ts_limited = ts[0:2000*60*60]
z_matrix1, bins = unit_utils.z_spike_matrix(ts_limited, pyr_units, binsize)

reducer1 = umap.UMAP(n_neighbors=n_neighbors)

embedding1 = reducer1.fit_transform(z_matrix1.T)
embedding1.shape
plt.scatter(
    embedding1[0:1000, 0],
    embedding1[0:1000, 1], marker='.')

subject = subjects.subjects["25-10"]

data = subjects.load_data(subject, units=True)

pyr_units = data['olm']['pyr_units']

binsize = 100
ts = data['olm']['Sleep1']['ts']

#
# only use first 60 minutes
#
ts_limited = ts[0:2000*60*30]
z_matrix2, bins = unit_utils.z_spike_matrix(ts_limited, pyr_units, binsize)



reducer2 = umap.UMAP(n_neighbors=n_neighbors)

embedding2 = reducer2.fit_transform(z_matrix2.T)
embedding2.shape
plt.scatter(
    embedding2[0:1000, 0],
    embedding2[0:1000, 1], marker='.')

print(embedding2)