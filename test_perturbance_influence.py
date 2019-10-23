import sys
import time
from multiprocessing import Pool, cpu_count
import numpy as np
from torch import nn, manual_seed

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
SAVE_PATH = './results/perturbation'
EVAL_F = 5
PCT = [[5, 5], [10, 10], [15, 15], [20, 20]]

EXPS = [{'type': 'Enc_Dec',  # Original
         'f_enc': [1, 5, 5, 5, 7, 10, 10],
         'n_enc': [256, 128, 64, 32, 16, 8, 4],
         'f_dec': [10, 10, 7, 5, 5, 5, 5],
         'n_dec': [4, 8, 16, 32, 64, 128, 256],
         'f_conv': [5, 5, 1],
         'ups': gc.WEI,
         'downs': gc.WEI},
        {'type': 'Enc_Dec',  # Original
         'f_enc': [1, 5, 5, 5, 7, 10, 10],
         'n_enc': [256]*7,
         'f_dec': [10, 10, 7, 5, 5, 5, 5],
         'n_dec': [256]*7,
         'f_conv': [5, 5, 1],
         'ups': None,
         'downs': None},
        {'type': 'AutoConv',
         'f_enc': [1, 2, 3, 4, 4],
         'kernel_enc': 10,
         'f_dec': [4, 4, 3, 2, 1],
         'kernel_dec': 10},
        {'type': 'AutoFC',
         'n_enc': [256, 1],
         'n_dec': [1, 256],
         'bias': True},
        {'type': 'Enc_Dec',  # Original
         'f_enc': [1, 5, 5, 7, 7, 10],
         'n_enc': [256, 64, 32, 16, 8, 4],
         'f_dec': [10, 7, 7, 5, 5, 5],
         'n_dec': [4, 8, 16, 32, 64, 256],
         'f_conv': [5, 5, 1],
         'ups': gc.WEI,
         'downs': gc.WEI},
        {'type': 'Enc_Dec',  # Original
         'f_enc': [1, 5, 5, 7, 7, 10],
         'n_enc': [256]*6,
         'f_dec': [10, 7, 7, 5, 5, 5],
         'n_dec': [256]*6,
         'f_conv': [5, 5, 1],
         'ups': None,
         'downs': None},
        {'type': 'Enc_Dec',  # HalfWeigths
         'f_enc': [1, 3, 3, 3, 3],
         'n_enc': [256, 64, 16, 8, 4],
         'f_dec': [3, 3, 3, 3, 3],
         'n_dec': [4, 8, 16, 64, 256],
         'f_conv': [3, 3, 1],
         'ups': gc.WEI,
         'downs': gc.WEI},
        {'type': 'AutoConv',
         'f_enc': [1, 3, 3, 3, 4],
         'kernel_enc': 8,
         'f_dec': [4, 3, 3, 3, 1],
         'kernel_dec': 8},
        {'type': 'AutoFC',
         'n_enc': [256, 1],
         'n_dec': [1, 256],
         'bias': False}]

# # Experiments with 64 nodes
# EXPS = [{'type': 'Enc_Dec',  # 192
#          'f_enc': [1, 4, 5, 5],
#          'n_enc': [64, 16, 8, 4],
#          'f_dec': [5, 5, 4, 4],
#          'n_dec': [4, 8, 16, 64],
#          'f_conv': [4, 4, 1],
#          'ups': gc.WEI,
#          'downs': gc.WEI},
#         {'type': 'Enc_Dec',  # 192
#          'f_enc': [1, 4, 5, 5],
#          'n_enc': [64]*4,
#          'f_dec': [5, 5, 4, 4],
#          'n_dec': [64]*4,
#          'f_conv': [4, 4, 1],
#          'ups': None,
#          'downs': None},
#         {'type': 'AutoConv',  # 210
#          'f_enc': [1, 2, 2, 3, 3],
#          'kernel_enc': 5,
#          'f_dec': [3, 3, 2, 2, 1],
#          'kernel_dec': 5},
#         {'type': 'AutoFC',  # 193
#          'n_enc': [64, 1],
#          'n_dec': [1, 64],
#          'bias': True},
#         {'type': 'Enc_Dec',  # 132
#          'f_enc': [1, 3, 3, 3, 3],
#          'n_enc': [64, 32, 16, 8, 4],
#          'f_dec': [3, 3, 3, 3, 3],
#          'n_dec': [4, 8, 16, 32, 64],
#          'f_conv': [3, 3, 1],
#          'ups': gc.WEI,
#          'downs': gc.WEI},
#         {'type': 'Enc_Dec',  # 132
#          'f_enc': [1, 3, 3, 3, 3],
#          'n_enc': [64]*5,
#          'f_dec': [3, 3, 3, 3, 3],
#          'n_dec': [64]*5,
#          'f_conv': [3, 3, 1],
#          'ups': None,
#          'downs': None},
#         {'type': 'Enc_Dec',  # 102
#          'f_enc': [1, 3, 3, 3],
#          'n_enc': [64, 16, 8, 4],
#          'f_dec': [3, 3, 3, 3],
#          'n_dec': [4, 8, 16, 64],
#          'f_conv': [3, 3, 1],
#          'ups': gc.WEI,
#          'downs': gc.WEI},
#         {'type': 'AutoConv',  # 140
#          'f_enc': [1, 2, 2, 2, 2],
#          'kernel_enc': 5,
#          'f_dec': [2, 2, 2, 2, 1],
#          'kernel_dec': 5},
#         {'type': 'AutoFC',  # 128
#          'n_enc': [64, 1],
#          'n_dec': [1, 64],
#          'bias': False}]


N_EXPS = len(EXPS)


def run(id, Gs, signals, lrn, pct):
    Gx, Gy = ds.perturbated_graphs(Gs['params'], pct[0], pct[1], pct=Gs['pct'],
                                   seed=SEED)
    data = ds.LinearDS2GS(Gx, Gy, signals['samples'], signals['L'],
                          signals['deltas'], median=signals['median'],
                          same_coeffs=signals['same_coeffs'])
    data.to_unit_norm()
    data.add_noise(signals['noise'], test_only=signals['test_only'])
    median_dist = np.median(np.linalg.norm(data.train_X-data.train_Y, axis=1))
    print('Signal {}: distance {}'.format(id, median_dist))
    data.to_tensor()

    epochs = 0
    mean_err = np.zeros(N_EXPS)
    med_err = np.zeros(N_EXPS)
    mse = np.zeros(N_EXPS)
    for i, exp in enumerate(EXPS):
        if exp['type'] == 'Linear':
            model = LinearModel(exp['N'])
        elif exp['type'] == 'Enc_Dec':
            clust_x = gc.MultiResGraphClustering(Gx, exp['n_enc'],
                                                 k=exp['n_enc'][-1],
                                                 up_method=exp['downs'])
            clust_y = gc.MultiResGraphClustering(Gy, exp['n_dec'],
                                                 k=exp['n_enc'][-1],
                                                 up_method=exp['ups'])
            net = GraphEncoderDecoder(exp['f_enc'], clust_x.sizes, clust_x.Ds,
                                      exp['f_dec'], clust_y.sizes, clust_y.Us,
                                      exp['f_conv'], As_dec=clust_y.As,
                                      As_enc=clust_x.As, act_fn=lrn['af'],
                                      last_act_fn=lrn['laf'], ups=exp['ups'],
                                      downs=exp['downs'])

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
        epochs, _, _ = model.fit(data.train_X, data.train_Y, data.val_X, data.val_Y)
        mean_err[i], med_err[i], mse[i] = model.test(data.test_X, data.test_Y)
        print('G: {}, {}-{} ({}): epochs {} - mse {} - MedianErr: {}'
              .format(id, i, exp['type'], model.count_params(), epochs,
                      mse[i], med_err[i]))

    return mean_err, med_err, mse


if __name__ == '__main__':
    # Set seeds
    np.random.seed(SEED)
    manual_seed(SEED)

    # Graphs parameters
    Gs = {}
    Gs['n_graphs'] = 15
    G_params = {}
    G_params['type'] = ds.SBM
    # G_params['N'] = N = 64
    # G_params['k'] = k = 4
    # G_params['p'] = [0.7, 0.8, 0.7, 0.8]
    # G_params['q'] = [[0, 0.05, 0.01, 0.0],
    #                  [0.05, 0, 0.01, 0.05],
    #                  [0.01, 0.01, 0, 0.05],
    #                  [0, 0.05, 0.05, 0]]

    G_params['N'] = N = 256
    G_params['k'] = k = 4
    G_params['p'] = 0.20
    G_params['q'] = [[0, 0.0075, 0, 0.0],
                     [0.0075, 0, 0.004, 0.0025],
                     [0, 0.004, 0, 0.005],
                     [0, 0.0025, 0.005, 0]]

    G_params['type_z'] = ds.RAND
    Gs['params'] = G_params
    Gs['pct'] = True
    Gs['pct_val'] = PCT

    # Signals
    signals = {}
    signals['L'] = 6
    signals['samples'] = [2000, 1000, 1000]
    signals['deltas'] = k
    signals['noise'] = 0
    signals['median'] = True
    signals['same_coeffs'] = False
    signals['test_only'] = True

    learning = {}
    learning['laf'] = nn.Tanh()
    learning['af'] = nn.Tanh()
    learning['lr'] = 0.01
    learning['dr'] = 0.9
    learning['batch'] = 10
    learning['epochs'] = 100
    learning['non_dec'] = 10

    start_time = time.time()
    mean_err = np.zeros((Gs['n_graphs'], N_EXPS, len(PCT)))
    median_err = np.zeros((Gs['n_graphs'], N_EXPS, len(PCT)))
    mse = np.zeros((Gs['n_graphs'], N_EXPS, len(PCT)))
    for i, pct in enumerate(PCT):
        print('PCT:', pct)
        if Gs['n_graphs'] > 1:
            with Pool(processes=N_CPUS) as pool:
                results = []
                for j in range(Gs['n_graphs']):
                    results.append(pool.apply_async(run,
                                   args=[j, Gs, signals, learning, pct]))
                for j in range(Gs['n_graphs']):
                    mean_err[j, :, i], median_err[j, :, i], mse[j, :, i] = \
                        results[j].get()
        else:
            mean_err[0, :, i], median_err[0, :, i], mse[0, :, i] = \
                run(0, Gs, signals, learning, pct)

    # Print summary
        utils.print_partial_results(pct, EXPS, mean_err[:, :, i],
                                    median_err[:, :, i])

    end_time = time.time()
    utils.print_results(PCT, EXPS, mean_err, median_err)
    print('Time: {} hours'.format((end_time-start_time)/3600))

    if SAVE:
        utils.save(SAVE_PATH, EXPS, mean_err, median_err, Gs, signals,
                   learning)
