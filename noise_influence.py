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


PATH = './results/noise/'
FILE_PREF = 'noise_'


SEED = 15
N_CPUS = cpu_count()
VERBOSE = False
SAVE = True
EVAL_F = 5
P_N = [0, 0.05]  # [0, .025, .05, 0.075, .1]


# Different size: 30 nodos menos
EXPS = [
        {'type': 'Enc_Dec',  # 2610
         'f_enc': [1, 15, 15, 15, 15, 15],
         'n_enc': [256, 64, 32, 16, 8, 4],
         'f_dec': [15, 15, 15, 15, 15, 15],
         'n_dec': [4, 8, 16, 32, 64, 226],
         'f_conv': [15, 15, 1],
         'ups': gc.WEI,
         'downs': gc.WEI,
         'early_stop': True,
         'fmt': 'o-'},
        {'type': 'Enc_Dec',  # 2610
         'f_enc': [1, 15, 15, 15, 15, 15],
         'n_enc': [256, 64, 32, 16, 8, 4],
         'f_dec': [15, 15, 15, 15, 15, 15],
         'n_dec': [4, 8, 16, 32, 64, 226],
         'f_conv': [15, 15, 1],
         'ups': gc.GF,
         'downs': gc.GF,
         'early_stop': True,
         'fmt': 'o--'},
        # {'type': 'Enc_Dec',  # 162
        #  'f_enc': [1, 3, 3, 3, 3, 3],
        #  'n_enc': [256, 64, 32, 16, 8, 4],
        #  'f_dec': [3, 3, 3, 3, 3, 3],
        #  'n_dec': [4, 8, 16, 32, 64, 226],
        #  'f_conv': [3, 3, 1],
        #  'ups': gc.WEI,
        #  'downs': gc.WEI,
        #  'early_stop': True,
        #  'fmt': 'o--'},

        {'type': 'AutoFC',  # 2641
         'n_enc': [256, 5],
         'n_dec': [5, 226],
         'bias': True,
         'early_stop': True,
         'fmt': 'X-'},
        # {'type': 'AutoFC',  # 709
        #  'n_enc': [256, 1],
        #  'n_dec': [1, 226],
        #  'bias': True,
        #  'early_stop': True,
        #  'fmt': 'X--'},

        {'type': 'AutoConv',
         'f_enc': [1, 6, 6, 6, 7, 7],
         'kernel_enc': 11,
         'f_dec': [7, 6, 1],
         'kernel_dec': 11,
         'early_stop': True,
         'fmt': 'P-'},
        #  {'type': 'AutoConv',
        #  'f_enc': [1, 1, 1, 1, 1, 2, 2, 2, 2],
        #  'kernel_enc': 6,
        #  'f_dec': [2, 2, 1],
        #  'kernel_dec': 6,
        #  'early_stop': True,
        #  'fmt': 'P--'}

        # {'type': 'Enc_Dec',  # Original
        #  'f_enc': [1, 5, 5, 5, 5, 5, 5],
        #  'n_enc': [256, 128, 64, 32, 16, 8, 4],
        #  'f_dec': [5, 5, 5, 5, 5, 5, 5],
        #  'n_dec': [4, 8, 16, 32, 64, 128, 226],
        #  'f_conv': [5, 5, 1],
        #  'ups': gc.WEI,
        #  'downs': gc.WEI,
        #  'early_stop': False,
        #  'fmt': 'X-'},

        # {'type': 'AutoConv',
        #  'f_enc': [1, 1, 3, 3, 3, 3],
        #  'kernel_enc': 11,      # Resulting size: Prev_size - kernel + 1
        #  'f_dec': [3, 2, 1],
        #  'kernel_dec': 11},
        # {'type': 'Enc_Dec',  # HalfWeigths
        #  'f_enc': [1, 3, 5, 5, 5, 5],
        #  'n_enc': [256, 128, 64, 16, 8, 4],
        #  'f_dec': [5, 5, 5, 5, 3, 3],
        #  'n_dec': [4, 8, 16, 64, 128, 226],
        #  'f_conv': [3, 3, 1],
        #  'ups': gc.WEI,
        #  'downs': gc.WEI},
        # {'type': 'AutoConv',
        #  'f_enc': [1, 2, 2, 2, 2, 3],
        #  'kernel_enc': 11,
        #  'f_dec': [3, 2, 1],
        #  'kernel_dec': 11},
        # {'type': 'Enc_Dec',  # HalfWeigths
        #  'f_enc': [1, 3, 3, 3, 3],
        #  'n_enc': [256, 64, 16, 8, 4],
        #  'f_dec': [3, 3, 3, 3, 3],
        #  'n_dec': [4, 8, 16, 64, 226],
        #  'f_conv': [3, 3, 1],
        #  'ups': gc.WEI,
        #  'downs': gc.WEI},
        # {'type': 'AutoConv',
        #  'f_enc': [1, 1, 1, 1, 1, 2, 2, 2, 2],
        #  'kernel_enc': 6,
        #  'f_dec': [2, 2, 1],
        #  'kernel_dec': 6}
         ]


N_EXPS = len(EXPS)


def run(id, Gs, Signals, lrn, p_n):
    if Gs['params']['type'] == ds.SBM:
        Gx, Gy = ds.nodes_perturbated_graphs(Gs['params'], Gs['pert'], seed=SEED)
    elif Gs['params']['type'] == ds.BA:
        Gx = ds.create_graph(Gs['params'], SEED)
        G_params_y = Gs['params'].copy()
        G_params_y['N'] = Gs['params']['N'] - Gs['p_n']
        Gy = ds.create_graph(G_params_y, 2*SEED)
    else:
        raise RuntimeError("Choose a valid graph type")
    data = ds.LinearDS2GSNodesPert(Gx, Gy, Signals['samples'], Signals['L'],
                                   Signals['deltas'], median=Signals['median'],
                                   same_coeffs=Signals['same_coeffs'])
    data.to_unit_norm()
    data.add_noise(p_n, test_only=Signals['test_only'])
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
        print('G: {}, {}-{} ({}): epochs {} - mse {} - TestErr: {}'
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
    if G_params['type'] == ds.SBM:
        G_params['k'] = k = 4
        G_params['p'] = 0.25
        G_params['q'] = [[0, 0.0075, 0, 0.0],
                        [0.0075, 0, 0.004, 0.0025],
                        [0, 0.004, 0, 0.005],
                        [0, 0.0025, 0.005, 0]]
        G_params['type_z'] = ds.RAND
    elif G_params['type'] == ds.BA:
        G_params['m0'] = 1
        G_params['m'] = 1
        k = int(N/32)   # In this case k is only used for the number of deltas
    else:
        raise RuntimeError("Choose a valid graph type")
    Gs['params'] = G_params
    Gs['pert'] = 30

    # G_params_y = {}
    # G_params_y['type'] = G_params['type']
    # G_params_y['N'] = G_params['N'] - 30
    # G_params_y['k'] = G_params['k']
    # G_params_y['p'] = G_params['p']
    # G_params_y['q'] = G_params['q']
    # G_params_y['type_z'] = G_params['type_z']
    # Gs['params_y'] = G_params_y

    # Signals
    Signals = {}
    Signals['L'] = 6
    Signals['samples'] = [2000, 1000, 1000]
    Signals['deltas'] = k
    Signals['noise'] = P_N
    Signals['median'] = True
    Signals['same_coeffs'] = False
    Signals['neg_coeffs'] = False
    Signals['test_only'] = True

    Net = {}
    Net['laf'] = nn.Tanh()
    Net['af'] = nn.Tanh()
    Net['lr'] = 0.001  # 0.001
    Net['dr'] = 1  # 0.9 
    Net['batch'] = 50  # 10
    Net['epochs'] = 50
    Net['non_dec'] = 10

    start_time = time.time()
    median_err = np.zeros((len(P_N), Gs['n_graphs'], N_EXPS))
    node_err = np.zeros((len(P_N), Gs['n_graphs'], N_EXPS))
    for i, p_n in enumerate(P_N):
        print('Noise:', p_n)
        with Pool(processes=N_CPUS) as pool:
            results = []
            for j in range(Gs['n_graphs']):
                results.append(pool.apply_async(run,
                               args=[j, Gs, Signals, Net, p_n]))
            for j in range(Gs['n_graphs']):
                params, median_err[i, j, :], node_err[i, j, :] = \
                    results[j].get()

        # Print summary
        utils.print_partial_results(p_n, EXPS, node_err[i, :, :],
                                    median_err[i, :, :])
    end_time = time.time()
    utils.print_results(P_N, EXPS, node_err, median_err)
    print('Time: {} hours'.format((end_time-start_time)/3600))
    legend = create_legend(params)
    fmts = [exp['fmt'] for exp in EXPS]
    utils.plot_results(median_err, P_N, legend=legend, fmts=fmts)
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
            'fmts': fmts,
            'legend': legend,
        }
        utils.save_results(FILE_PREF, PATH, data)
