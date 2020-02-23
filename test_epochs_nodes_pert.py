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


PATH = './results/epochs/'
FILE_PREF = 'epochs_'

SEED = 15
N_CPUS = cpu_count()
VERBOSE = False
SAVE = False
SAVE_PATH = './results/node_pert'
EVAL_F = 1

MAX_EPOCHS = 100

EXPS = [
        {'type': 'Enc_Dec',  # 2610
         'f_enc': [1, 15, 15, 15, 15, 15],
         'n_enc': [256, 64, 32, 16, 8, 4],
         'f_dec': [15, 15, 15, 15, 15, 15],
         'n_dec': [4, 8, 16, 32, 64, 226],
         'f_conv': [15, 15, 1],
         'ups': gc.WEI,
         'downs': gc.WEI,
         'fmt': ['o-', 'o--']},
        {'type': 'Enc_Dec',  # 2610
         'f_enc': [1, 15, 15, 15, 15, 15],
         'n_enc': [256, 64, 32, 16, 8, 4],
         'f_dec': [15, 15, 15, 15, 15, 15],
         'n_dec': [4, 8, 16, 32, 64, 226],
         'f_conv': [15, 15, 1],
         'ups': gc.GF,
         'downs': gc.GF,
         'fmt': ['o-', 'o--']},
        # {'type': 'Enc_Dec',  # 162
        #  'f_enc': [1, 3, 3, 3, 3, 3],
        #  'n_enc': [256, 64, 32, 16, 8, 4],
        #  'f_dec': [3, 3, 3, 3, 3, 3],
        #  'n_dec': [4, 8, 16, 32, 64, 226],
        #  'f_conv': [3, 3, 1],
        #  'ups': gc.WEI,
        #  'downs': gc.WEI,
        #  'early_stop': True,
        #  'fmt': ['X-', 'X--']},

        # {'type': 'Enc_Dec',  # 2610
        #  'f_enc': [1, 30, 30, 30, 30, 30],
        #  'n_enc': [256, 64, 32, 16, 8, 4],
        #  'f_dec': [30, 30, 30, 30, 30, 30],
        #  'n_dec': [4, 8, 16, 32, 64, 226],
        #  'f_conv': [30, 1],
        #  'ups': gc.WEI,
        #  'downs': gc.WEI,
        #  'fmt': ['o-', 'o--']},

        # {'type': 'AutoConv',  # 
        #  'f_enc': [1, 5, 6, 6, 6],
        #  'kernel_enc': 13,
        #  'f_dec': [6, 6, 5, 5, 1],
        #  'kernel_dec': 13,
        #  'early_stop': True,
        #  'fmt': 'P-',
        #  'fmt': ['o-', 'o--']},
        {'type': 'AutoFC',  # 2641
         'n_enc': [256, 5],
         'n_dec': [5, 226],
         'bias': True,
         'fmt': ['v-', 'v--']},
        # {'type': 'AutoFC',
        #  'n_enc': [256, 1],
        #  'n_dec': [1, 226],
        #  'bias': True,
        #  'fmt': ['^-', '^--']}


        ]
N_EXPS = len(EXPS)


def run(id, Gs, Signals, Lrn):
    Gx, Gy = ds.nodes_perturbated_graphs(Gs['params'], Gs['pert'], seed=SEED)
    # Create Signals
    data = ds.LinearDS2GSNodesPert(Gx, Gy, Signals['samples'], Signals['L'],
                                   Signals['deltas'], median=Signals['median'],
                                   same_coeffs=Signals['same_coeffs'])
    data.to_unit_norm()
    data.add_noise(Signals['noise'], test_only=Signals['test_only'])
    data.to_tensor()

    epochs = 0
    train_err = np.zeros((N_EXPS, MAX_EPOCHS))
    val_err = np.zeros((N_EXPS, MAX_EPOCHS))
    params = np.zeros(N_EXPS)
    test_err = np.zeros(N_EXPS)
    for i, exp in enumerate(EXPS):
        if exp['type'] is 'Enc_Dec':
            clust_x = gc.MultiResGraphClustering(Gx, exp['n_enc'],
                                                k=exp['n_enc'][-1],
                                                up_method=exp['downs'])
            clust_y = gc.MultiResGraphClustering(Gy, exp['n_dec'],
                                                k=exp['n_enc'][-1],
                                                up_method=exp['ups'])
            net = GraphEncoderDecoder(exp['f_enc'], clust_x.sizes, clust_x.Ds,
                                    exp['f_dec'], clust_y.sizes, clust_y.Us,
                                    exp['f_conv'], As_dec=clust_y.As,
                                    As_enc=clust_x.As, act_fn=Lrn['af'],
                                    last_act_fn=Lrn['laf'], ups=exp['ups'],
                                    downs=exp['downs'])
        elif exp['type'] == 'AutoFC':
            net = FCAutoencoder(exp['n_enc'], exp['n_dec'], bias=exp['bias'])
        elif exp['type'] == 'AutoConv':
            net = ConvAutoencoder(exp['f_enc'], exp['kernel_enc'],
                                  exp['f_dec'], exp['kernel_dec'])
        else:
            raise RuntimeError('Unknown exp type')
        model = Model(net, learning_rate=Lrn['lr'], decay_rate=Lrn['dr'],
                      batch_size=Lrn['batch'], epochs=MAX_EPOCHS,
                      eval_freq=EVAL_F, max_non_dec=Lrn['non_dec'],
                      early_stop=Lrn['early_stop'], verbose=VERBOSE)
    
 
        params[i] = model.count_params()
        epochs, train_err[i, :], val_err[i, :] = model.fit(data.train_X, data.train_Y,
                                                           data.val_X, data.val_Y)
        _, test_err[i], mse = model.test(data.test_X, data.test_Y)
        print('G: {}, {}-{} ({}): epochs {} - mse {} - MedianErr: {}'
              .format(id, i, exp['type'], params[i], epochs,
                      mse, test_err[i]))
    # Multiplying by N because we're interested in the error of the whole
    # signal, # not only the node error
    return train_err.T*Gy.N, val_err.T*Gy.N, params, test_err


if __name__ == '__main__':
    # Set random seed
    np.random.seed(SEED)
    manual_seed(SEED)

    # Graph parameters
    Gs = {}
    Gs['n_graphs'] = 15
    G_params = {}
    G_params['type'] = ds.SBM
   
    G_params['N'] = N = 256
    G_params['k'] = k = 4
    G_params['p'] = 0.25
    G_params['q'] = [[0, 0.0075, 0, 0.0],
                     [0.0075, 0, 0.004, 0.0025],
                     [0, 0.004, 0, 0.005],
                     [0, 0.0025, 0.005, 0]]
    G_params['type_z'] = ds.RAND
    Gs['params'] = G_params
    Gs['pert'] = 30

    # Signals
    Signals = {}
    Signals['L'] = 6
    Signals['samples'] = [2000, 1000, 1000]
    Signals['deltas'] = k
    Signals['noise'] = 0
    Signals['median'] = True
    Signals['same_coeffs'] = False
    Signals['test_only'] = True
    
    # NOTA: los valores pequeños parecen haber funcionado bien en la convergencia!
    # lr: 0.0001 dr: 0.9 --> merece la pena quitar Tanh?
    Net = {}
    Net['laf'] = None  # nn.Tanh()
    Net['af'] = nn.Tanh()
    Net['lr'] =  0.001
    Net['dr'] = 0.75  # 1
    Net['batch'] = 10
    Net['epochs'] = 100  # 50 --> experimento actual!!
    Net['non_dec'] = 15
    Net['early_stop'] = False
    Net['steps'] = int(Signals['samples'][0]/Net['batch'])

    print("CPUs used:", N_CPUS)
    start_time = time.time()
    train_err = np.zeros((MAX_EPOCHS, Gs['n_graphs'], N_EXPS))
    val_err = np.zeros((MAX_EPOCHS, Gs['n_graphs'], N_EXPS))
    test_err = np.zeros((Gs['n_graphs'], N_EXPS))
    with Pool(processes=N_CPUS) as pool:
        results = []
        for j in range(Gs['n_graphs']):
            results.append(pool.apply_async(run,
                           args=[j, Gs, Signals, Net]))
        for j in range(Gs['n_graphs']):
            train_err[:, j, :], val_err[:, j, :], params, test_err[j, :] = results[j].get()

    end_time = time.time()
    print('--- {} hours ---'.format((time.time()-start_time)/3600))
    epochs = np.arange(MAX_EPOCHS)
    # Plot median of all realiztions
    utils.plot_overfitting(train_err, val_err, params, show=True)
    utils.print_partial_results(0, EXPS, test_err, test_err)
    # Plot only first realization
    first_t_err = train_err[:, 0, :].reshape([MAX_EPOCHS, 1, N_EXPS])
    first_v_err = val_err[:, 0, :].reshape([MAX_EPOCHS, 1, N_EXPS])
    utils.plot_overfitting(first_t_err, first_v_err, params)
    if SAVE:
        data = {
            'seed': SEED,
            'exps': EXPS,
            'Gs': Gs,
            'Signals': Signals,
            'Net': Net,
            'node_err': None,
            'train_err': train_err,
            'val_err': val_err,
            'params': params,
        }
        utils.save_results(FILE_PREF, PATH, data)
