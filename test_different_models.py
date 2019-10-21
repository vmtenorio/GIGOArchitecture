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
SAVE_PATH = './results/diff_models'
EVAL_F = 5
P_N = [0, .05]  # [0, .05, .1]

EXPS = [{'type': 'Linear', 'N': 64},
        {'type': 'Enc_Dec',  # Constant
         'f_enc': [1, 3, 3, 3, 3],
         'n_enc': [64, 32, 16, 8, 4],
         'f_dec': [3, 3, 3, 3, 3],
         'n_dec': [4, 8, 16, 32, 64],
         'f_conv': [3, 3, 1],
         'ups': gc.WEI,
         'downs': gc.WEI},
        {'type': 'Enc_Dec',  # Constant
         'f_enc': [1, 3, 3, 3, 3],
         'n_enc': [64, 32, 16, 8, 4],
         'f_dec': [3, 3, 3, 3, 3],
         'n_dec': [4, 8, 16, 32, 64],
         'f_conv': [3, 3, 1],
         'ups': gc.BIN,
         'downs': gc.BIN},
        {'type': 'Enc_Dec',  # Constant
         'f_enc': [1, 3, 3, 3, 3],
         'n_enc': [64, 32, 16, 8, 4],
         'f_dec': [3, 3, 3, 3, 3],
         'n_dec': [4, 8, 16, 32, 64],
         'f_conv': [3, 3, 1],
         'ups': gc.NO_A,
         'downs': gc.NO_A},
        {'type': 'Enc_Dec',  # Constant
         'f_enc': [1, 3, 3, 3, 3],
         'n_enc': [64]*5,
         'f_dec': [3, 3, 3, 3, 3],
         'n_dec': [64]*5,
         'f_conv': [3, 3, 1],
         'ups': None,
         'downs': None},
        {'type': 'AutoConv',
         'f_enc': [1, 3, 3, 4],
         'kernel_enc': 3,
         'f_dec': [4, 3, 3, 1],
         'kernel_dec': 3},
        {'type': 'AutoFC',
         'n_enc': [64, 1],
         'n_dec': [1, 64],
         'bias': True}]

N_EXPS = len(EXPS)


def run(id, Gs, signals, lrn, p_n):
    Gx, Gy = ds.perturbated_graphs(Gs['params'], Gs['pct_val'][0],
                                   Gs['pct_val'][1], pct=Gs['pct'],
                                   seed=SEED)
    data = ds.LinearDS2GS(Gx, Gy, signals['samples'], signals['L'],
                          signals['deltas'], median=signals['median'],
                          same_coeffs=signals['same_coeffs'])
    data.to_unit_norm()
    median_dist = np.median(np.linalg.norm(data.train_X-data.train_Y, axis=1))
    print('Signal {}: distance {}'.format(id, median_dist))
    data.add_noise(p_n, test_only=signals['test_only'])
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
    Gs['n_graphs'] = 25
    G_params = {}
    G_params['type'] = ds.SBM
    G_params['N'] = N = 64
    G_params['k'] = k = 4
    G_params['p'] = [0.6, 0.7, 0.6, 0.8]
    G_params['q'] = [[0, 0.05, 0.01, 0.0],
                     [0.05, 0, 0.01, 0.05],
                     [0.01, 0.01, 0, 0.05],
                     [0, 0.05, 0.05, 0]]
    G_params['type_z'] = ds.RAND
    Gs['params'] = G_params
    Gs['pct'] = True
    Gs['pct_val'] = [10, 10]

    # Signals
    signals = {}
    signals['L'] = 6
    signals['samples'] = [2000, 500, 500]
    signals['deltas'] = k
    signals['noise'] = P_N
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
    mean_err = np.zeros((Gs['n_graphs'], N_EXPS, len(P_N)))
    median_err = np.zeros((Gs['n_graphs'], N_EXPS, len(P_N)))
    mse = np.zeros((Gs['n_graphs'], N_EXPS, len(P_N)))
    for i, p_n in enumerate(P_N):
        print('P_N:', p_n)
        with Pool(processes=N_CPUS) as pool:
            results = []
            for j in range(Gs['n_graphs']):
                results.append(pool.apply_async(run,
                               args=[j, Gs, signals, learning, p_n]))
            for j in range(Gs['n_graphs']):
                mean_err[j, :, i], median_err[j, :, i], mse[j, :, i] = \
                    results[j].get()

        # Print summary
        utils.print_partial_results(p_n, EXPS, mean_err[:, :, i],
                                    median_err[:, :, i])

        utils.save(SAVE_PATH, EXPS, mean_err[:, :, i], median_err[:, :, i],
                   Gs, signals, learning)

    end_time = time.time()
    utils.print_results(P_N, EXPS, mean_err, median_err)
    print('Time: {} hours'.format((end_time-start_time)/3600))

    if SAVE:
        utils.save(SAVE_PATH, EXPS, mean_err, median_err, Gs, signals,
                   learning)
