# file to run hierarchical clustering for one subject example

import matplotlib.pyplot as plt
import numpy as np

import unit_utils

import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import subjects
from scipy.cluster.hierarchy import dendrogram
from sklearn.neighbors import NearestCentroid


distance = 50

subject = subjects.subjects["18-1"]

data = subjects.load_data(subject, units=True)

pyr_units = data['olm']['pyr_units']

binsize = 100
ts = data['olm']['Sleep1']['ts']

#
# only use first 60 minutes
#
ts_limited = ts[0:2000*60*60]
z_matrix, bins = unit_utils.z_spike_matrix(ts_limited, pyr_units, binsize)


clustering = AgglomerativeClustering(n_clusters=None,distance_threshold=distance)


y_predict = clustering.fit_predict(z_matrix.T)
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



plt.title("Hierarchical Clustering Dendrogram")
# plot dendrogram

plot_dendrogram(clustering, truncate_mode="lastp", p=clustering.n_clusters_)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


clf = NearestCentroid()
clf.fit(z_matrix.T, y_predict)
sns.clustermap(clf.centroids_.T,row_cluster=False)
plt.show()

