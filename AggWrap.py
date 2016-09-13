from __future__ import print_function, division
import numpy as np
from .ClusterWrap import _ClusterWrap
from sklearn.cluster import AgglomerativeClustering

class AggWrap(_ClusterWrap):
    def __init__(self, points_matrix, points_names = None, n_clusters = 0, **kwargs):
        super(self.__class__, self).__init__(points_matrix, points_names)
        #initialize other variables
        self.n_clusters = None
        self.agg_cluster = None
        if n_clusters in range(1, len(self.points_names)+1):
            self.perform_agg_cluster(n_clusters, **kwargs)

    def perform_agg_cluster(self, n_clusters, **kwargs):
        self.n_clusters = n_clusters
        self.agg_cluster = AgglomerativeClustering(n_clusters = n_clusters, **kwargs)
        self.clust_labels = self.agg_cluster.fit_predict(self.points_matrix)
        clust_dict = dict()
        for lab, point in [(l,e) for l,e in zip(self.clust_labels, self.points_names)]:
            try:
                clust_dict[lab].append(point)
            except KeyError:
                clust_dict[lab] = [point]
        self.clust_dict = clust_dict
        clust_cents = dict()
        clust_vars = dict()
        tot_var = 0.0
        for key in self.clust_dict.keys():
            points = self.clust_dict[key]
            N = len(points)
            idxs = [self.points_idx_dict[point] for point in points]
            point_vecs = [(self.points_matrix)[idx, :] for idx in idxs]
            cent = np.sum(point_vecs, axis = 0) / N
            var = sum([np.linalg.norm(vec - cent)**2 for vec in point_vecs])
            tot_var += var
            clust_cents[key] = cent
            clust_vars[key] = var / N
        self.tot_var = tot_var / len(self.points_names)
        self.clust_cents = clust_cents
        self.clust_vars = clust_vars

        print("%d agglomerative clusters made." % n_clusters)
        print("Execute: self.clust_show() to print out clusters.")
        print("")
