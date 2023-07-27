# file to run hierarchical clustering for one subject example

import matplotlib.pyplot as plt
import numpy as np

import unit_utils

import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import subjects
from scipy.cluster.hierarchy import dendrogram
from sklearn.neighbors import NearestCentroid
import umap.plot
from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler


# try parameters
distances = [5,7,9,11]
binsizes = [10, 20, 50, 100]
subs = ['18-1']

for s in subs:

    subject = subjects.subjects[s]

    data = subjects.load_data(subject, units=True)

    pyr_units = data['olm']['pyr_units']

    ts = data['olm']['Sleep1']['ts']

    #
    # only use first 60 minutes
    #
    ts_limited = ts[0:2000 * 60 * 60]
    for distance in distances:
        for binsize in binsizes:



            s_matrix, bins = unit_utils.spike_matrix(ts_limited, pyr_units, binsize)

            # convert to sparse
            scaled_matrix = MaxAbsScaler().fit_transform(s_matrix.T)

            unique_m = np.unique(scaled_matrix, axis=0)

            sparse_matrix = sparse.csr_matrix(unique_m)




            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance)


            y_predict = clustering.fit_predict(unique_m)
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


            plt.figure()
            plt.title("Hierarchical Clustering Dendrogram")
            # plot dendrogram

            plot_dendrogram(clustering, truncate_mode="lastp", p=clustering.n_clusters_)
            plt.xlabel("Number of points in node (or index of point if no parenthesis).")
            plt.savefig(fname="../../output/hierarchical/dendrogram{}_{}_{}.png".format(s, distance, binsize),bbox_inches='tight', dpi=300)


            """clf = NearestCentroid()
            clf.fit(unique_m, y_predict)
            sns.clustermap(clf.centroids_.T,row_cluster=False)
            plt.show()"""



            # plot umap transform
            n_neighbors = 10
            reducer1 = umap.UMAP(n_neighbors=n_neighbors)




            embedding1 = reducer1.fit_transform(sparse_matrix)
            embedding1.shape
            #plt.scatter(
            #    embedding1[0:1000, 0],
            #    embedding1[0:1000, 1], marker='.')
            plt.figure()
            umap.plot.points(reducer1, labels=y_predict)
            plt.savefig(fname="../../output/hierarchical/umap{}_{}_{}.png".format(s, distance, binsize), bbox_inches='tight', dpi=300)
