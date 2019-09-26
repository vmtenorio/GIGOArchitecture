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
    # TODO: test hier_A function
    # TODO: test build_arch
    # Graph parameters
    n_graphs = 25
    G_params = {}
    G_params['type'] = data_sets.SBM #SBM or ER
    G_params['N']  = N = 128
    G_params['k']  = k = 4
    G_params['p'] = 0.4
    G_params['q'] = 0.01
    G_params['type_z'] = data_sets.CONT
    eps_c = 0.001
    eps_d = 0.1

    # Signal parameters
    L = 6
    n_samples = [2000,200,200]
    
    # Encoder size
    feat_enc = [1, 5, 10, 10] #[1, 3, 3, 3]
    nodes_enc = [N, 32, 16, k]

    # Decoder size
    feat_dec =  [10, 10, 5, 5] #[3, 3, 3, 3]
    nodes_dec = [k, 16, 32, N]
    # Layers of only convolutions
    feat_only_conv = [5, 3, 1]
    ups = gc.WEI

    """
    # Smaller graphs!
    G_params = {}
    G_params['type'] = data_sets.SBM #SBM or ER
    G_params['N']  = N = 64
    G_params['k']  = k = 4
    G_params['p'] = 0.6
    G_params['q'] = 0.01
    G_params['type_z'] = data_sets.RAND
    eps_c = 0.0005
    eps_d = 0.1
    # Encoder size
    feat_enc = [1, 5, 10, 10] #[1, 3, 3, 3]
    nodes_enc = [N, 16, 8, k]
    # Decoder size
    feat_dec =  [10, 10, 5, 5] #[3, 3, 3, 3]
    nodes_dec = [k, 8, 16, N]
    # Layers of only convolutions
    feat_only_conv = [5, 3, 1]
    """

    start_time = time.time()
    mean_err = np.zeros(n_graphs)
    mse = np.zeros(n_graphs)
    for i in range(n_graphs):
        # Create graphs
        Gx, Gy = data_sets.perturbated_graphs(G_params, eps_c, eps_d, seed=SEED)
        
        # Create graph signals
        data = data_sets.DiffusedSparse2GS(Gx, Gy, n_samples, L, G_params['k'],
                                        median=True)
        data.to_unit_norm()
        data.to_tensor()

        # Obtein clusters
        cluster_x = gc.MultiResGraphClustering(Gx, nodes_enc, k=4, up_method=None)
        cluster_y = gc.MultiResGraphClustering(Gy, nodes_dec, k=4, up_method=ups)

        # _, axes = plt.subplots(1, 2)
        # axes[0].spy(Gx.A)
        # axes[1].spy(Gy.A)
        # cluster_x.plot_labels(False)
        # cluster_y.plot_labels()
        # sys.exit()

        net = architecture.GraphEncoderDecoder(feat_enc, cluster_x.sizes,
                                                cluster_x.Ds, feat_dec,
                                                cluster_y.sizes, cluster_y.Us,
                                                feat_only_conv, As_dec=cluster_y.As)

        if i==0:
            print('Signal Size:', N,'Params:', net.count_params())
        net.fit(data.train_X, data.train_Y, data.val_X, data.val_Y, decay_rate=.9,
                n_epochs=100, batch_size=100, lr=0.1, verbose=True)
        mean_err[i], mse[i] = net.test(data.test_X, data.test_Y)
        print('Graph {}: Mean MSE: {} - Mean Err: {}'.format(i, mse[i], mean_err[i]))

    end_time = time.time()
    print('Time: {} minutes'.format((end_time-start_time)/60))
    print('Mean MSE: {} - Mean Err: {}'.format(np.mean(mse), np.mean(mean_err)))

