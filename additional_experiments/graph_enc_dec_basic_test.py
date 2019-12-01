import sys
import os
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, manual_seed

sys.path.insert(0, 'graph_enc_dec')
from graph_enc_dec import data_sets
from graph_enc_dec import graph_clustering as gc
from graph_enc_dec import architecture
from graph_enc_dec.model import Model, LinearModel
from graph_enc_dec.standard_architectures import ConvAutoencoder, FCAutoencoder


SEED = 15
N_CPUS = cpu_count()

# Graph parameters
pct = True
# creat = 0.0005
# dest = 0.05
creat = 5
dest = 5

# Learning parameters
lr = 1e-3
dr = 0.9
bs = 10
epochs = 100

# Signal parameters
deltas = 4
median = True
same_coeffs = True
p_n = 0


def estimate_signals(id, G_params, n_samples, L, nodes_enc,
                     nodes_dec, ups, feat_enc, feat_dec, feat_only_conv):
    # Create graphs
    Gx, Gy = data_sets.perturbated_graphs(G_params, creat, dest, pct=pct,
                                          seed=SEED)
    diff_links = np.sum(Gx.A != Gy.A)/2/Gx.Ne*100
    print('Links different(%):', diff_links)

    # Create graph signals
    data = data_sets.NonLinearDS2GS(Gx, Gy, n_samples, L, deltas,
                                    median=median, same_coeffs=same_coeffs)
    data.to_unit_norm()
    data.add_noise(p_n, test_only=True)
    mean_dist = np.median(np.linalg.norm(data.train_X-data.train_Y, axis=1))
    print('Distance signals:', mean_dist)
    data.to_tensor()

    N = G_params['N']
    k = G_params['k']

    model = LinearModel(N)
    model.fit(data.train_X, data.train_Y, data.val_X, data.val_Y)
    mean_err, med_err, mse = model.test(data.test_X, data.test_Y)
    print('LINEAR Graph {}: N: {} Mean MSE: {} - Mean Err: {} - Median Err: {}'
          .format(id, Gx.N, mse, mean_err, med_err))

    # Obtein clusters
    cluster_x = gc.MultiResGraphClustering(Gx, nodes_enc, k=k, up_method=ups)
    cluster_y = gc.MultiResGraphClustering(Gy, nodes_dec, k=k, up_method=ups)


    # Graph Autoenc
    net = architecture.GraphEncoderDecoder(feat_enc, cluster_x.sizes,
                                           cluster_x.Ds, feat_dec,
                                           cluster_y.sizes, cluster_y.Us,
                                           feat_only_conv, As_dec=cluster_y.As,
                                           As_enc=cluster_x.As, ups=ups,
                                           last_act_fn=nn.Tanh(), downs=ups,
                                           act_fn=nn.Tanh())
    model = Model(net, decay_rate=dr, epochs=epochs, batch_size=bs,
                  learning_rate=lr, verbose=False, eval_freq=1, max_non_dec=10)
    print('Model parameters: ', model.count_params())
    iters, _, _ = model.fit(data.train_X, data.train_Y, data.val_X, data.val_Y)

    mean_err, med_err, mse = model.test(data.test_X, data.test_Y)
    print('G: {}, ({}): epochs {} - mse {} - MedianErr: {}'
          .format(id, model.count_params(), iters,
                  mse, med_err))
    return mean_err, mse, med_err, diff_links, mean_dist, iters


if __name__ == '__main__':
    # TODO: test hier_A function
    # TODO: test build_arch

    np.random.seed(SEED)
    manual_seed(SEED)

    # Signal parameters
    L = 6
    n_samples = [5000, 1000, 1000]

    # Graph parameters
    n_graphs = 25
    G_params = {}
    G_params['type'] = data_sets.SBM # SBM or ER
    G_params['N'] = N = 256
    G_params['k'] = k = 4
    G_params['p'] = 0.20
    G_params['q'] = 0.015/4
    G_params['type_z'] = data_sets.RAND

    # Model parameters
    feat_enc = [1, 5, 5, 10, 10, 15, 15]
    nodes_enc = [256, 128, 64, 32, 16, 8, 4]
    feat_dec = [15, 15, 10, 10, 5, 5, 5]
    nodes_dec = [4, 8, 16, 32, 64, 128, 256]
    feat_only_conv = [5, 5, 1]

    # feat_enc = [1, 3, 3, 6]
    # nodes_enc = [N, 64, 16, k]
    # feat_dec = [6, 3, 3, 3]
    # nodes_dec = [k, 16, 64, N]
    # feat_only_conv = [3, 3, 1]
    ups = gc.WEI

    start_time = time.time()
    mean_err = np.zeros(n_graphs)
    med_err = np.zeros(n_graphs)
    mse = np.zeros(n_graphs)
    links = np.zeros(n_graphs)
    dist = np.zeros(n_graphs)
    iters = np.zeros(n_graphs)
    if N_CPUS > 1:
        with Pool(processes=N_CPUS) as pool:
            results = []
            for i in range(n_graphs):
                results.append(pool.apply_async(estimate_signals,
                        args=[i, G_params, n_samples, L, nodes_enc,
                              nodes_dec, ups, feat_enc, feat_dec, feat_only_conv]))
            for i in range(n_graphs):
                mean_err[i], mse[i], med_err[i], links[i], dist[i], iters[i] = \
                      results[i].get()
    else:
        print('Not multithread')
        for i in range(n_graphs):
            mean_err[i], mse[i], med_err[i], links[i], dist[i], iters[i] = \
                  estimate_signals(i, G_params,
                                   n_samples, L, nodes_enc, nodes_dec, ups,
                                   feat_enc, feat_dec, feat_only_conv)

    end_time = time.time()
    print('Time: {} minutes'.format((end_time-start_time)/60))
    print('Median link diff: {} - Median signal dist: {} - Median iters: {}'
          .format(np.median(links), np.median(dist), np.median(iters)))
    print('MSE: Mean: {} - Median: {} - STD: {}'
          .format(np.mean(mse), np.median(mse), np.std(mean_err)))
    print('Mean Err: Mean: {} - Median: {} - STD: {}'
          .format(np.mean(mean_err), np.median(mean_err), np.std(mean_err)))
    print('Median Err: Mean: {} - Median: {} - STD: {}'
          .format(np.mean(med_err), np.median(med_err), np.std(med_err)))
