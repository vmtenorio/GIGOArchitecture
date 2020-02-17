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

PATH = './results/links_pert/'
FILE_PREF = 'links_'

SEED = 15
N_CPUS = cpu_count()
VERBOSE = False
SAVE = True
EVAL_F = 5
PCT = [[5, 5], [10, 10], [15, 15], [20, 20]]

EXPS = [
        {'type': 'Enc_Dec',  # 2610
         'f_enc': [1, 15, 15, 15, 15, 15],
         'n_enc': [256, 64, 32, 16, 8, 4],
         'f_dec': [15, 15, 15, 15, 15, 15],
         'n_dec': [4, 8, 16, 32, 64, 256],
         'f_conv': [15, 15, 1],
         'ups': gc.WEI,
         'downs': gc.WEI,
         'early_stop': True,
         'fmt': 'o-'},
        {'type': 'Enc_Dec',  # 162
         'f_enc': [1, 3, 3, 3, 3, 3],
         'n_enc': [256, 64, 32, 16, 8, 4],
         'f_dec': [3, 3, 3, 3, 3, 3],
         'n_dec': [4, 8, 16, 32, 64, 256],
         'f_conv': [3, 3, 1],
         'ups': gc.WEI,
         'downs': gc.WEI,
         'early_stop': True,
         'fmt': 'o--'},

        {'type': 'AutoFC',  # 2641
         'n_enc': [256, 5],
         'n_dec': [5, 256],
         'bias': True,
         'early_stop': True,
         'fmt': 'X-'},
        {'type': 'AutoFC',  # 709
         'n_enc': [256, 1],
         'n_dec': [1, 256],
         'bias': True,
         'early_stop': True,
         'fmt': 'X--'},

         {'type': 'AutoConv',  # 
         'f_enc': [1, 5, 6, 6, 6],
         'kernel_enc': 13,
         'f_dec': [6, 6, 5, 5, 1],
         'kernel_dec': 13,
         'early_stop': True,
         'fmt': 'P-'},
         {'type': 'AutoConv',  # 
         'f_enc': [1, 2, 3, 3],
         'kernel_enc': 5,
         'f_dec': [3, 3, 2, 1],
         'kernel_dec': 5,
         'early_stop': True,
         'fmt': 'P--'}

        # {'type': 'Enc_Dec',  # Original
        #  'f_enc': [1, 5, 5, 5, 7, 10, 10],
        #  'n_enc': [256, 128, 64, 32, 16, 8, 4],
        #  'f_dec': [10, 10, 7, 5, 5, 5, 5],
        #  'n_dec': [4, 8, 16, 32, 64, 128, 256],
        #  'f_conv': [5, 5, 1],
        #  'ups': gc.WEI,
        #  'downs': gc.WEI},
        # {'type': 'Enc_Dec',  # Original
        #  'f_enc': [1, 5, 5, 5, 7, 10, 10],
        #  'n_enc': [256]*7,
        #  'f_dec': [10, 10, 7, 5, 5, 5, 5],
        #  'n_dec': [256]*7,
        #  'f_conv': [5, 5, 1],
        #  'ups': None,
        #  'downs': None},
        # {'type': 'AutoConv',
        #  'f_enc': [1, 2, 3, 4, 4],
        #  'kernel_enc': 10,
        #  'f_dec': [4, 4, 3, 2, 1],
        #  'kernel_dec': 10},
        # {'type': 'AutoFC',
        #  'n_enc': [256, 1],
        #  'n_dec': [1, 256],
        #  'bias': True},
        # {'type': 'Enc_Dec',  # Original
        #  'f_enc': [1, 5, 5, 7, 7, 10],
        #  'n_enc': [256, 64, 32, 16, 8, 4],
        #  'f_dec': [10, 7, 7, 5, 5, 5],
        #  'n_dec': [4, 8, 16, 32, 64, 256],
        #  'f_conv': [5, 5, 1],
        #  'ups': gc.WEI,
        #  'downs': gc.WEI},
        # {'type': 'Enc_Dec',  # Original
        #  'f_enc': [1, 5, 5, 7, 7, 10],
        #  'n_enc': [256]*6,
        #  'f_dec': [10, 7, 7, 5, 5, 5],
        #  'n_dec': [256]*6,
        #  'f_conv': [5, 5, 1],
        #  'ups': None,
        #  'downs': None},
        # {'type': 'Enc_Dec',  # HalfWeigths
        #  'f_enc': [1, 3, 3, 3, 3],
        #  'n_enc': [256, 64, 16, 8, 4],
        #  'f_dec': [3, 3, 3, 3, 3],
        #  'n_dec': [4, 8, 16, 64, 256],
        #  'f_conv': [3, 3, 1],
        #  'ups': gc.WEI,
        #  'downs': gc.WEI},
        # {'type': 'AutoConv',
        #  'f_enc': [1, 3, 3, 3, 4],
        #  'kernel_enc': 8,
        #  'f_dec': [4, 3, 3, 3, 1],
        #  'kernel_dec': 8},
        # {'type': 'AutoFC',
        #  'n_enc': [256, 1],
        #  'n_dec': [1, 256],
        #  'bias': False}
        ]

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


def run(id, Gs, Signals, lrn, pct):
    Gx, Gy = ds.perturbated_graphs(Gs['params'], pct[0], pct[1], pct=Gs['pct'],
                                   perm=True, seed=SEED)
    data = ds.LinearDS2GSLinksPert(Gx, Gy, Signals['samples'], Signals['L'],
                                   Signals['deltas'], median=Signals['median'],
                                   same_coeffs=Signals['same_coeffs'])
    data.to_unit_norm()
    data.add_noise(Signals['noise'], test_only=Signals['test_only'])
    median_dist = np.median(np.linalg.norm(data.train_X-data.train_Y, axis=1))
    print('Signal {}: distance {}'.format(id, median_dist))
    data.to_tensor()

    epochs = 0
    params = np.zeros(N_EXPS)
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
                          verbose=VERBOSE, early_stop=exp['early_stop'])
        epochs, _, _ = model.fit(data.train_X, data.train_Y, data.val_X, data.val_Y)
        _, med_err[i], mse[i] = model.test(data.test_X, data.test_Y)
        params[i] = model.count_params()
        print('G: {}, {}-{} ({}): epochs {} - mse {} - MedianErr: {}'
              .format(id, i, exp['type'], params[i], epochs,
                      mse[i], med_err[i]))

    return params, med_err, mse


def create_legend(params):
    legend = []
    for i, exp in enumerate(EXPS):
        txt = ''
        if exp['type'] is 'Enc_Dec':
            txt = 'G-E/D-{}: Ups: {}, Down: {}, Stop: {}'
            txt = txt.format(params[i], exp['ups'], exp['downs'], exp['early_stop'])
        elif exp['type'] is 'AutoFC':
            txt = 'AE-FC-{}, Stop: {}'.format(params[i], exp['early_stop'])
        elif exp['type'] is 'AutoConv':
            txt = 'AE-CV-{}, Stop: {}'.format(params[i], exp['early_stop'])
        legend.append(txt)
    return legend


if __name__ == '__main__':
    # Set seeds
    np.random.seed(SEED)
    manual_seed(SEED)

    # Graphs parameters
    Gs = {}
    Gs['n_graphs'] = 15
    G_params = {}
    G_params['type'] = ds.SBM
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
    Signals = {}
    Signals['L'] = 6
    Signals['samples'] = [2000, 1000, 1000]
    Signals['deltas'] = k
    Signals['noise'] = 0
    Signals['median'] = True
    Signals['same_coeffs'] = False
    Signals['test_only'] = True

    Net = {}
    Net['laf'] = nn.Tanh()
    Net['af'] = nn.Tanh()
    Net['lr'] = 0.01
    Net['dr'] = 0.9
    Net['batch'] = 10
    Net['epochs'] = 50
    Net['non_dec'] = 10

    start_time = time.time()
    median_err = np.zeros((len(PCT), Gs['n_graphs'], N_EXPS))
    node_err = np.zeros((len(PCT), Gs['n_graphs'], N_EXPS))
    for i, pct in enumerate(PCT):
        print('PCT:', pct)
        with Pool(processes=N_CPUS) as pool:
            results = []
            for j in range(Gs['n_graphs']):
                results.append(pool.apply_async(run,
                               args=[j, Gs, Signals, Net, pct]))
            for j in range(Gs['n_graphs']):
                params, median_err[i, j, :], node_err[i, j, :] = \
                    results[j].get()

        # Print summary
        utils.print_partial_results(pct, EXPS, node_err[i, :, :],
                                    median_err[i, :, :])

    end_time = time.time()
    utils.print_results(PCT, EXPS, node_err, median_err)
    print('Time: {} hours'.format((end_time-start_time)/3600))
    legend = create_legend(params)
    fmts = [exp['fmt'] for exp in EXPS]
    PCT_sum = np.array(PCT).sum(axis=1)
    utils.plot_results(median_err, PCT_sum, legend=legend, fmts=fmts)
    if SAVE:
        data = {
            'seed': SEED,
            'exps': EXPS,
            'Gs': Gs,
            'Signals': Signals,
            'Net': Net,
            'node_err': node_err,
            'err': median_err,
            'params': params,
        }
        utils.save_results(FILE_PREF, PATH, data)
