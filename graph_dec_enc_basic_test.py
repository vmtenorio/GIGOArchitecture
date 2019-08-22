import sys
import os
import time
sys.path.insert(0, 'graph_enc_dec')

import graph_enc_dec
from graph_enc_dec import utils
from graph_enc_dec import graph_clustering as gc
from graph_enc_dec import architecture

SEED = 15

if __name__ == '__main__':
    # Graph parameters
    G_params = {}
    G_params['type'] = utils.SBM #SBM or ER
    G_params['N']  = 10
    G_params['k']  = 2
    G_params['p'] = 0.7
    G_params['q'] = 0.05

    # Initia and final values are assumed to be 1
    feat_d = [2, 2, 2, 3]
    feat_u = [3, 2, 2, 2]
    nodes_d = [10, 5, 2]
    nodes_u = [2, 5, 10]
    ups = gc.WEI

    start_time = time.time()
    Gx = utils.create_graph(G_params, seed=SEED, type_z=utils.ALT)
    Gy = utils.create_graph(G_params, seed=SEED, type_z=utils.ALT)
    print("Gx and Gy TRUE node labesl:")
    print(Gx.info['node_com'])
    print(Gy.info['node_com'])
    
    # For Gx
    print("Clustering Gx:")
    cluster_x = gc.MultiRessGraphClustering(Gx, nodes_d, k=2)
    # Remember, the structure is the same but the data is obtained differently
    # The same sturcture can be used
    cluster_x.compute_hierarchy_ascendance()
    print(cluster_x.clusters_size)
    print(cluster_x.labels)
    print(cluster_x.ascendance)
    cluster_x.plot_dendrogram()
    cluster_x.compute_hierarchy_A(ups)
    
    print("Clustering Gy:")
    cluster_y = gc.MultiRessGraphClustering(Gx, nodes_u, k=2)
    cluster_y.plot_labels()
    cluster_y.compute_hierarchy_descendance()
    print(cluster_y.clusters_size)
    print(cluster_y.labels)
    print(cluster_y.descendance)
    cluster_y.plot_dendrogram()

    net = architecture.GraphEncoderDecoder(feat_d, feat_u, nodes_d, nodes_u,
                                            cluster_x.ascendance, 
                                            cluster_y.descendance, cluster_x.hier_A,
                                            cluster_y.hier_A, ups, ups)





