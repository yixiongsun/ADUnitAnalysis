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
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

import subjects

# Try PCA transformation

subject = subjects.subjects["25-10"]

data = subjects.load_data(subject, units=True)

pyr_units = data['olm']['pyr_units']

binsize = 10
ts = data['olm']['Sleep1']['ts']
z_matrix, bins = unit_utils.z_spike_matrix(ts, pyr_units, binsize)




ripple_windows = data['olm']['Sleep1']['ripple_windows']
r_bins, r_start_bins = unit_utils.ripple_bins(bins,ripple_windows,ts)

r_matrix = unit_utils.ripple_spike_matrix(ts, ripple_windows, pyr_units)
sns.heatmap(r_matrix)
plt.show()


# try sleep2

binsize = 10
ripple_windows = data['olm']['Sleep2']['ripple_windows']
ts2 = data['olm']['Sleep2']['ts']
r_matrix_2 = unit_utils.ripple_spike_matrix(ts2, ripple_windows, pyr_units)

z_ripple = z_matrix[:,r_bins]

pca = PCA()

# concate the two for fit
pca.fit(np.concatenate((r_matrix.T, r_matrix_2.T)))
#z_transformed = pca.fit_transform(z_matrix.T)
z_ripple_transformed = pca.transform(r_matrix.T)

z_ripple_transformed2 = pca.transform(r_matrix_2.T)

#z_transformed = pca.transform(z_matrix.T)


#z_ripple_transformed = pca.transform(z_ripple.T)


# try isomaps
#iso = Isomap(n_neighbors=8)
#iso.fit(z_ripple.T)
#z_manifold = iso.transform(z_ripple.T)


#t = np.arange(z_manifold.shape[0])
#plt.scatter(z_manifold[:, 0], z_manifold[:, 1], c=t)
#z_manifold_all = iso.transform(z_matrix.T)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#ax.scatter(z_transformed[:,0], z_transformed[:,1], z_transformed[:,2], marker='.', s=4)
ax.scatter(z_ripple_transformed[:,0], z_ripple_transformed[:,1], z_ripple_transformed[:,2], marker='.', s=4)
ax.scatter(z_ripple_transformed2[:,0], z_ripple_transformed2[:,1], z_ripple_transformed2[:,2], marker='.', s=4)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.figure()
#plt.scatter(z_transformed[:,0], z_transformed[:,1],  marker='.', s=4)
#t= np.arange(0,z_ripple_transformed.shape[0])
plt.scatter(z_ripple_transformed[:,0], z_ripple_transformed[:,1],  marker='.', s=4)#, c=t)
plt.scatter(z_ripple_transformed2[:,0], z_ripple_transformed2[:,1],  marker='.', s=4)#, c=t)

plt.colorbar()
# try other pcs
plt.show()


plt.figure()
length = z_ripple_transformed.shape[0]
#plt.scatter(z_transformed[:,0], z_transformed[:,1],  marker='.', s=4)
#t= np.arange(0,z_ripple_transformed.shape[0])
plt.scatter(z_ripple_transformed[:,0], z_ripple_transformed[:,1],  marker='.', s=4, c="Blue")#, c=t)
#plt.figure()
plt.scatter(z_ripple_transformed2[:,0], z_ripple_transformed2[:,1],  marker='.', s=4, c="Red")#, c=t)
plt.scatter(z_ripple_transformed2[0:length,0], z_ripple_transformed2[0:length:,1],  marker='.', s=4,c='Orange')
# try other pcs
plt.show()