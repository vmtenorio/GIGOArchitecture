import sys
import os
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

sys.path.insert(0, 'graph_enc_dec')
from graph_enc_dec import data_sets
from graph_enc_dec import graph_clustering as gc
from graph_enc_dec import architecture
from graph_enc_dec.model import Model
from graph_enc_dec.standard_architectures import ConvAutoencoder


SEED = 15
N_CPUS = cpu_count()


def estimate_signals(i, G_params, eps_c, eps_d, n_samples, L, nodes_enc, nodes_dec,
                     ups, feat_enc, feat_dec, feat_only_conv):
    # Create graphs
    Gx, Gy = data_sets.perturbated_graphs(G_params, eps_c, eps_d, seed=SEED)

    # Create graph signals
    data = data_sets.LinearDS2GS(Gx, Gy, n_samples, L, 3*G_params['k'],
                                    median=True)
    data.to_unit_norm()
    print('Median Diff between Y and X:', np.median(np.linalg.norm((data.train_X-data.train_Y)**2,1)))

    X = data.train_X
    Beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(data.train_Y)
    test_Y = data.test_Y
    est_Y_test = data.test_X.dot(Beta)
    test_err = np.sum((est_Y_test-test_Y)**2, axis=1)/np.linalg.norm(data.test_Y)**2
    print('Linear model: mean err: {} - median: {}'.format(np.mean(test_err), np.median(test_err)))

    data.to_tensor()

    # Obtein clusters
    cluster_x = gc.MultiResGraphClustering(Gx, nodes_enc, k=4, up_method=None)
    cluster_y = gc.MultiResGraphClustering(Gy, nodes_dec, k=4, up_method=ups)

    # Standar ConvAutoenc
    net = architecture.GraphEncoderDecoder(feat_enc, [Gx.N]*7,
                                           cluster_x.Ds, feat_dec,
                                           [Gx.N]*7, cluster_y.Us,
                                           feat_only_conv, As_dec=cluster_y.As,
                                           last_act_fn=nn.Tanh(), act_fn=nn.Tanh())
    model = Model(net, decay_rate=.9, epochs=25, batch_size=100, learning_rate=0.05,
                  verbose=True, eval_freq=1, max_non_dec=5)
    print('Model parameters: ', model.count_params())
    model.fit(data.train_X, data.train_Y, data.val_X, data.val_Y)
    mean_err, median_err, mse = model.test(data.test_X, data.test_Y)
    print('Autoencoder: Graph {}: N: {} Mean MSE: {} - Mean Err: {} - Median Err: {}'
          .format(i, Gx.N, mse, mean_err, median_err))

    # Graph Autoenc
    net = architecture.GraphEncoderDecoder(feat_enc, cluster_x.sizes,
                                           cluster_x.Ds, feat_dec,
                                           cluster_y.sizes, cluster_y.Us,
                                           feat_only_conv, As_dec=cluster_y.As,
                                           last_act_fn=nn.Tanh(), act_fn=nn.Tanh())
    model = Model(net, decay_rate=.9, epochs=25, batch_size=100, learning_rate=0.05,
                  verbose=True, eval_freq=1, max_non_dec=5)
    print('Model parameters: ', model.count_params())
    model.fit(data.train_X, data.train_Y, data.val_X, data.val_Y)
    mean_err, median_err, mse = model.test(data.test_X, data.test_Y)
    print('GRAPH ENC-DEC Graph {}: N: {} Mean MSE: {} - Mean Err: {} - Median Err: {}'
            .format(i, Gx.N, mse, mean_err, median_err))
    return mean_err, mse, model.count_params()


if __name__ == '__main__':
    np.random.seed(SEED)
    # TODO: test hier_A function
    # TODO: test build_arch

    # Signal parameters
    L = 5
    n_samples = [5000, 1000, 1000]

    # Graph parameters
    n_graphs = 1  # 25

    G_params = {}
    G_params['type'] = data_sets.SBM # SBM or ER
    G_params['N'] = N = 64  # 256
    G_params['k'] = k = 4
    G_params['p'] = 0.6  # 0.15
    G_params['q'] = 0.01  # 0.01/4
    G_params['type_z'] = data_sets.RAND
    eps_c = 0.0005
    eps_d = 0.05

    # Model parameters
    feat_enc = [1, 5, 5, 10, 10, 15, 20]
    nodes_enc = [N, 32, 32, 16, 16, 8, k]  # [N, 128, 64, 32, 16, 8, k]
    feat_dec = [20, 15, 10, 10, 5, 5, 5]
    nodes_dec = [k, 8, 16, 16, 32, 32, N]  # [k, 8, 16, 32, 64, 124, N]
    feat_only_conv = [5, 5, 1]
    ups = gc.WEI

    start_time = time.time()
    mean_err = np.zeros(n_graphs)
    mse = np.zeros(n_graphs)
    if n_graphs > 1:
        with Pool(processes=N_CPUS) as pool:
            results = []
            for i in range(n_graphs):
                results.append(pool.apply_async(estimate_signals,
                        args=[i, G_params, eps_c, eps_d, n_samples, L, nodes_enc,
                              nodes_dec, ups, feat_enc, feat_dec, feat_only_conv]))
            for i in range(n_graphs):
                mean_err[i], mse[i], params = results[i].get()
    else:
        mean_err[0], mse[0], params = estimate_signals(0,G_params,eps_c, eps_d,
                                            n_samples,L,nodes_enc,nodes_dec,ups,
                                            feat_enc,feat_dec,feat_only_conv)

    end_time = time.time()
    print('Time: {} minutes'.format((end_time-start_time)/60))
    print('Mean MSE: {} - Mean Err: {} - Median Err: {} - STD: {}'
          .format(np.mean(mse), np.mean(mean_err), np.median(mean_err), np.std(mean_err)))