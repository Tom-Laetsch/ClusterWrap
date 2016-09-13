from __future__ import print_function, division
import numpy as np
from sklearn.cluster import AgglomerativeClustering

class _ClusterWrap(object):
    def __init__(self, points_matrix, points_names = None):
        self.points_matrix = points_matrix
        #create default point names if none given
        if points_names == None:
            r,c = (self.points_matrix).shape
            points_names = []
            for i in range(r):
                points_names.append("POINT_%s" % i)
        self.points_names = points_names
        #create dictionary pointing point names to matrix row
        points_idx_dict = dict()
        for idx, point in enumerate(self.points_names):
            points_idx_dict[point] = idx
        self.points_idx_dict = points_idx_dict
        #initialize other variables
        self.clust_labels = None
        #keys are cluster label, values are point names in cluster
        self.clust_dict = dict()
        #centroid of each cluster
        self.clust_cents = dict()
        #variance in a each cluster
        self.clust_vars = dict()
        #total variance (of points from respective centers)
        self.tot_var = None

    def clust_show(self, mode = '', msg = 'Model Clustering:'):
        #First create a clustered dictionary for ease of use
        print(msg)
        for key in self.clust_dict.keys():
            var = self.clust_vars[key]
            print("-- Cluster: %s -- Cluster Variance: %.4f --" % (key,var))
            if mode == 'long':
                std = np.sqrt(var)
                for point in (self.clust_dict)[key]:
                    vec = (self.points_matrix)[self.points_idx_dict[point],:]
                    cent = self.clust_cents[key]
                    dist = np.linalg.norm(vec - cent)
                    if dist > 0 and std > 0:
                        z = dist / std
                    else:
                        z = 0.0
                    global_z = dist / np.sqrt(self.tot_var)
                    print("\t%s:" % point)
                    print("\t --dist to center: %.4f" % dist)
                    print("\t --local z score: %.4f" % z)
                    print("\t --global z score: %.4f" %  global_z)
                print("\n")
            elif mode == 'short':
                for point in self.clust_dict[key]:
                    print(point, end="  ")
                print("\n")
            else:
                for point in (self.clust_dict)[key]:
                    vec = (self.points_matrix)[self.points_idx_dict[point],:]
                    cent = self.clust_cents[key]
                    dist = np.linalg.norm(vec - cent)
                    print("\t%s -- dist to center: %.4f" % (point, dist))
                print("\n")

    def predict(self, vec):
        keys = list(self.clust_cents.keys())
        idx_min = 0
        dist_min = np.linalg.norm(vec - self.clust_cents[keys[idx_min]])
        for idx in range(1, len(keys)):
            dist = np.linalg.norm(vec - self.clust_cents[keys[idx]])
            if dist < dist_min:
                dist_min = dist
                idx_min = idx
        return keys[idx_min], dist_min

    def find_atoms(self, atom_len = 1):
        atoms = []
        atoms_cnt = 0
        non_atoms_cnt = 0
        for key in self.clust_dict.keys():
            if len(self.clust_dict[key]) <= atom_len:
                 atoms += self.clust_dict[key]
                 atoms_cnt += 1
            else:
                non_atoms_cnt += 1

        print("%d atom clusters found; %d non-atom clusters found." % (atoms_cnt, non_atoms_cnt))
        return atoms, atoms_cnt, non_atoms_cnt
