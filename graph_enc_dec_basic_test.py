import sys
import os
import time
sys.path.insert(0, 'graph_enc_dec')

from graph_enc_dec import data_sets
from graph_enc_dec import graph_clustering as gc
from graph_enc_dec import architecture

import numpy as np
import matplotlib.pyplot as plt

SEED = 15

if __name__ == '__main__':
    np.random.seed(SEED)

    # Graph parameters
    G_params = {}
    G_params['type'] = data_sets.SBM #SBM or ER
    G_params['N']  = 10
    G_params['k']  = 2
    G_params['p'] = 0.8
    G_params['q'] = 0.2
    G_params['type_z'] = data_sets.CONT
    eps1 = 0.1
    eps2 = 0.3

    # Signal parameters
    L = 6
    n_samples = [1000,200,200]
    
    # Encoder size
    feat_enc = [1, 2, 3]
    nodes_enc = [10, 5, 2]

    # Decoder size
    feat_dec = [3, 2, 2]    #[3, 2, 2, 2, 1]
    nodes_dec = [2, 5, 10]  #[2, 5, 10, 10, 10]

    # Layers of only convolutions
    feat_only_conv = [2, 2, 1] # 
    ups = gc.WEI

    start_time = time.time()
    # Create graphs
    Gx, Gy = data_sets.perturbated_graphs(G_params, eps1, eps2, seed=SEED)
    
    #Gx.plot()
    #Gy.plot()
    #plt.show()

    # Create graph signals
    data = data_sets.DiffusedSparse2GS(Gx, Gy, n_samples, L, G_params['k'])
    data.to_unit_norm()
    data.to_tensor()

    # For Gx
    print("Clustering Gx:")
    cluster_x = gc.MultiResGraphClustering(Gx, nodes_enc, k=2, up_method=None)

    print("Clustering Gy:")
    cluster_y = gc.MultiResGraphClustering(Gy, nodes_dec, k=2, up_method=ups)

    #TODO: review hier_A function
    #TODO: test build_arch

    # Nodes_d and nodes_u should take the value from cluster!
    net = architecture.GraphEncoderDecoder(feat_enc, cluster_x.sizes,
                                            cluster_x.Ds, feat_dec,
                                            cluster_y.sizes, cluster_y.Us,
                                            feat_only_conv, As_dec=cluster_y.As)

    
    print('N params:', net.count_params())
    net.fit(data.train_X, data.train_Y, data.val_X, data.val_Y)




