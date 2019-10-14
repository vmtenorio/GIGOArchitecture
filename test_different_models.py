import sys
import time
from multiprocessing import Pool, cpu_count
import numpy as np
from torch import nn

sys.path.insert(0, 'graph_enc_dec')
from graph_enc_dec import data_sets as ds
from graph_enc_dec import graph_clustering as gc
from graph_enc_dec.architecture import GraphEncoderDecoder
from graph_enc_dec.model import Model, LinearModel
from graph_enc_dec.standard_architectures import ConvAutoencoder, FCAutoencoder
from graph_enc_dec import utils


SEED = 15
N_CPUS = cpu_count()
VERBOSE = False
SAVE = True
SAVE_PATH = './results/diff_models'
EVAL_F = 5
P_N = [0, .05]

EXPS = [{'type': 'Linear', 'N': 256},
        {'type': 'Enc_Dec',
         'f_enc': [1, 5, 5, 10, 10, 15, 15],
         'n_enc': [256, 128, 64, 32, 16, 8, 4],
         'f_dec': [15, 15, 10, 10, 5, 5, 5],
         'n_dec': [4, 8, 16, 32, 64, 124, 256],
         'f_conv': [5, 3, 1],
         'ups': gc.WEI},
        {'type': 'Enc_Dec',
         'f_enc': [1, 10, 10, 10],
         'n_enc': [256, 64, 16, 4],
         'f_dec': [10, 10, 10, 10],
         'n_dec': [4, 16, 64, 256],
         'f_conv': [10, 4, 1],
         'ups': gc.WEI},
        {'type': 'Enc_Dec',
         'f_enc': [1, 10, 10, 15, 15, 20, 20],
         'n_enc': [256, 128, 64, 32, 16, 8, 4],
         'f_dec': [20, 20, 15, 15, 10, 10, 10],
         'n_dec': [4, 8, 16, 32, 64, 124, 256],
         'f_conv': [10, 5, 1],
         'ups': gc.WEI},
        {'type': 'Enc_Dec',
         'f_enc': [1, 5, 5, 10, 10, 15, 15],
         'n_enc': [256]*7,
         'f_dec': [15, 15, 10, 10, 5, 5, 5],
         'n_dec': [256]*7,
         'f_conv': [5, 3, 1],
         'ups': gc.WEI}]

# EXPS = [{'type': 'Enc_Dec',
#          'f_enc': [1, 5, 5, 10, 10, 15, 15],
#          'n_enc': [256, 128, 64, 32, 16, 8, 4],
#          'f_dec': [15, 15, 10, 10, 5, 5, 5],
#          'n_dec': [4, 8, 16, 32, 64, 124, 256],
#          'f_conv': [5, 3, 1],
#          'ups': gc.WEI},
#         {'type': 'Enc_Dec',
#          'f_enc': [1, 5, 5, 10, 10, 15, 15],
#          'n_enc': [256]*7,
#          'f_dec': [15, 15, 10, 10, 5, 5, 5],
#          'n_dec': [256]*7,
#          'f_conv': [5, 3, 1],
#          'ups': gc.WEI},
#         {'type': 'Linear', 'N': 256},
#         {'type': 'AutoConv',
#          'f_enc': [1, 5, 5, 5],
#          'kernel_enc': 3,
#          'f_dec': [5, 5, 5, 1],
#          'kernel_dec': 3},
#         {'type': 'AutoFC',
#          'n_enc': [256, 3],
#          'n_dec': [3, 256],
#          'bias': True}]
N_EXPS = len(EXPS)


def test_models(id, Gs, signals, lrn, p_n):
    Gx, Gy = ds.perturbated_graphs(Gs['params'], Gs['eps_c'], Gs['eps_d'],
                                   seed=SEED)
    data = ds.LinearDS2GS(Gx, Gy, signals['samples'], signals['L'],
                          signals['deltas'], median=signals['median'])
    data.to_unit_norm()
    data.add_noise(p_n, test_only=False)
    data.to_tensor()

    mean_err = np.zeros(N_EXPS)
    med_err = np.zeros(N_EXPS)
    mse = np.zeros(N_EXPS)
    for i, exp in enumerate(EXPS):
        if exp['type'] == 'Linear':
            model = LinearModel(exp['N'])
        elif exp['type'] == 'Enc_Dec':
            clust_x = gc.MultiResGraphClustering(Gx, exp['n_enc'],
                                                 k=exp['n_enc'][-1],
                                                 up_method=exp['ups'])
            clust_y = gc.MultiResGraphClustering(Gy, exp['n_dec'],
                                                 k=exp['n_enc'][-1],
                                                 up_method=exp['ups'])
            net = GraphEncoderDecoder(exp['f_enc'], clust_x.sizes, clust_x.Ds,
                                      exp['f_dec'], clust_y.sizes, clust_y.Us,
                                      exp['f_conv'], As_dec=clust_y.As,
                                      act_fn=lrn['af'], last_act_fn=lrn['laf'])
        elif exp['type'] == 'AutoConv':
            net = ConvAutoencoder(exp['f_enc'], exp['kernel_enc'],
                                  exp['f_dec'], exp['kernel_dec'])
        elif exp['type'] == 'AutoFC':
            net = FCAutoencoder(exp['n_enc'], exp['n_dec'], bias=exp['bias'])
        else:
            raise RuntimeError('Unknown experiment type')
        if exp['type'] != 'Linear':
            model = Model(net, learning_rate=lrn['lr'], decay_rate=lrn['dr'],
                          batch_size=lrn['batch'], epochs=lrn['epochs'],
                          eval_freq=EVAL_F, max_non_dec=lrn['non_dec'],
                          verbose=VERBOSE)
        model.fit(data.train_X, data.train_Y, data.val_X, data.val_Y)
        mean_err[i], med_err[i], mse[i] = model.test(data.test_X, data.test_Y)
        print('G: {}, {}-{} ({}): mse {} - MeanErr {} - MedianErr: {}'
              .format(id, i, exp['type'], model.count_params(), mse[i],
                      mean_err[i], med_err[i]))

    return mean_err, med_err, mse


if __name__ == '__main__':
    # Graphs parameters
    Gs = {}
    Gs['n_graphs'] = 25
    G_params = {}
    G_params['type'] = ds.SBM  # SBM or ER
    G_params['N'] = N = 256
    G_params['k'] = k = 4
    G_params['p'] = 0.2
    G_params['q'] = 0.015/4
    G_params['type_z'] = ds.RAND
    Gs['params'] = G_params
    Gs['eps_c'] = 0.0005  # Equivalent to create 0.97% of links
    Gs['eps_d'] = 0.05   # Equivalent to destroy 5% of links

    # Signals
    signals = {}
    signals['L'] = 5
    signals['samples'] = [10000, 2000, 2000]
    signals['deltas'] = 3*k
    signals['noise'] = P_N
    signals['median'] = True

    learning = {}
    learning['laf'] = nn.Tanh()
    learning['af'] = nn.Tanh()
    learning['lr'] = 0.1
    learning['dr'] = 0.9
    learning['batch'] = 10
    learning['epochs'] = 100
    learning['non_dec'] = 10

    start_time = time.time()
    mean_err = np.zeros((Gs['n_graphs'], N_EXPS, len(P_N)))
    median_err = np.zeros((Gs['n_graphs'], N_EXPS, len(P_N)))
    mse = np.zeros((Gs['n_graphs'], N_EXPS, len(P_N)))
    for i, p_n in enumerate(P_N):
        print('P_N:', p_n)
        if Gs['n_graphs'] > 1:
            with Pool(processes=N_CPUS) as pool:
                results = []
                for j in range(Gs['n_graphs']):
                    results.append(pool.apply_async(test_models,
                                   args=[j, Gs, signals, learning, p_n]))
                for j in range(Gs['n_graphs']):
                    mean_err[j, :, i], median_err[j, :, i], mse[j, :, i] = \
                        results[j].get()
        else:
            mean_err[0, :, i], median_err[0, :, i], mse[0, :, i] = \
                test_models(0, Gs, signals, learning, p_n)

    # Print summary
        utils.print_partial_results(p_n, EXPS, mean_err[:, :, i],
                                    median_err[:, :, i])

        utils.save(SAVE_PATH, EXPS, mean_err, median_err, Gs, signals,
                   learning)

    end_time = time.time()
    utils.print_results(P_N, EXPS, mean_err, median_err)
    print('Time: {} hours'.format((end_time-start_time)/3600))

    if SAVE:
        utils.save(SAVE_PATH, EXPS, mean_err, median_err, Gs, signals,
                   learning)
