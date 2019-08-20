import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.sparse.csgraph import dijkstra
#from scipy import sparse
import matplotlib.pyplot as plt
from pygsp.graphs import Graph

class MultiRessGraphClustering():
    # k represents the size of the root cluster
    def __init__(self, G, n_clust, k, algorithm='spectral_clutering', 
                    method='maxclust', link_fun='average'):
        self.G = G
        # self.clusters_size = n_clust
        self.clusters_size = []
        self.cluster_alg = getattr(self, algorithm)
        self.k = k
        self.link_fun = link_fun
        self.labels = []
        self.Z = None
        self.descendance = {}
        self.ascendance = {}
        self.hier_A = []
        self.cluster_alg(G)
        
        for t in n_clust[:-1]:
            # t represent de relative distance, so it is necessary to obtain the 
            # real desired distance
            if method == 'distance':
                t = t*self.Z[-k,2]
            level_labels = fcluster(self.Z, t, criterion=method)
            self.labels.append(level_labels)
            self.clusters_size.append(np.unique(level_labels).size)
        self.labels.append(np.arange(1,G.W.shape[0]+1))
        self.clusters_size.append(G.W.shape[0])

    def distance_clustering(self, G):
        D = dijkstra(G.W)
        D = D[np.triu_indices_from(D,1)]
        self.Z = linkage(D,self.link_fun)

    def spectral_clutering(self, G):
        G.compute_laplacian()
        G.compute_fourier_basis()
        X = G.U[:,1:self.k]
        self.Z = linkage(X, self.link_fun)

    def plot_dendrogram(self):
        plt.figure()
        dendrogram(self.Z, orientation='left', no_labels=True)
        plt.gca().tick_params(labelsize=16)
        plt.show()

    def compute_hierarchy_descendance(self):
        # Maybe descendance should be the upsampling matrices
        for i in range(len(self.clusters_size)-1):
            self.descendance[i] = []
            # Find parent (labels i) of each child cluster (labels i+1)
            for j in range(self.clusters_size[i+1]):
                indexes = np.where(self.labels[i+1] == j+1)
                # Check if all has the same value!!!
                n_parents = np.unique(self.labels[i+1][indexes]).size
                if n_parents != 1:
                    raise RuntimeError("child {} belong to {} parents".format(j,n_parents))

                parent_id = self.labels[i][indexes][0]
                self.descendance[i].append(parent_id)

        return self.descendance

    def compute_hierarchy_ascendance(self):
        for i in range(len(self.clusters_size)-1,0,-1):
            self.ascendance[i] = []

    def compute_hierarchy_A(self, up_method):
        if up_method == 'no_A' or up_method == None or up_method == 'original':
            return

        A = self.G.W.todense()
        for i in range(len(self.clusters_size)):
            N = self.clusters_size[i]
            self.hier_A.append(np.zeros((N, N)))

            inter_clust_links = 0
            for j in range(N-1):
                nodes_c1 = np.where(self.labels[i] == j+1)[0]
                for k in range(j+1,N):
                    nodes_c2 = np.where(self.labels[i] == k+1)[0]
                    sub_A = A[nodes_c1,:][:,nodes_c2]

                    if up_method == 'binary' and np.sum(sub_A) > 0:
                        self.hier_A[i][j,k] = self.hier_A[i][k,j] = 1
                    if up_method == 'weighted':
                        self.hier_A[i][j,k] = np.sum(sub_A)
                        self.hier_A[i][k,j] = self.hier_A[i][j,k]
                        inter_clust_links += np.sum(sub_A)
            if up_method == 'weighted':
                self.hier_A[i] = self.hier_A[i]/inter_clust_links
        return self.hier_A

    def plot_labels(self, show=True):
        n_labels = len(self.labels)-1
        _, axes= plt.subplots(1, n_labels)
        self.G.set_coordinates()
        for i in range(n_labels):
            self.G.plot_signal(self.labels[i], ax=axes[i])
        if show:    
            plt.show()

    def plot_hier_A(self, show=True):
        _, axes = plt.subplots(2, len(self.hier_A))
        for i in range(len(self.hier_A)):
            G = Graph(self.hier_A[i])
            G.set_coordinates()
            axes[0,i].spy(self.hier_A[i])
            G.plot(ax=axes[1,i])
        if show:
            plt.show()