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
SAVE_PATH = './results/node_pert'
EVAL_F = 5
PERT = [10, 20, 30, 40, 50]

CONV1 = [{'f_enc': [1, 2, 2, 2, 3],  # 10
          'kernel_enc': 11,
          'f_dec': [3, 2, 2, 1],
          'kernel_dec': 11},
         {'f_enc': [1, 2, 2, 2, 3],  # 20
          'kernel_enc': 11,
          'f_dec': [3, 3, 1],
          'kernel_dec': 11},
         {'f_enc': [1, 2, 2, 2, 2, 3],  # 30
          'kernel_enc': 11,
          'f_dec': [3, 2, 1],
          'kernel_dec': 11},
         {'f_enc': [1, 2, 2, 2, 3, 3],  # 40
          'kernel_enc': 11,
          'f_dec': [3, 1],
          'kernel_dec': 11},
         {'f_enc': [1, 1, 1, 1, 2, 2, 2, 3],  # 50
          'kernel_enc': 11,
          'f_dec': [3, 2, 1],
          'kernel_dec': 11}]

CONV2 = [{'f_enc': [1, 1, 1, 2, 2],  # 10
          'kernel_enc': 11,
          'f_dec': [2, 1, 1, 1],
          'kernel_dec': 11},
         {'f_enc': [1, 1, 1, 2, 2],  # 20
          'kernel_enc': 11,
          'f_dec': [2, 1, 1],
          'kernel_dec': 11},
         {'f_enc': [1, 1, 2, 2, 2],  # 30
          'kernel_enc': 11,
          'f_dec': [2, 1],
          'kernel_dec': 11},
         {'f_enc': [1, 1, 1, 1, 2, 2],  # 40
          'kernel_enc': 11,
          'f_dec': [2, 1],
          'kernel_dec': 11},
         {'f_enc': [1, 1, 1, 1, 1, 2, 2],  # 50
          'kernel_enc': 11,
          'f_dec': [2, 1],
          'kernel_dec': 11}]


EXPS = [{'type': 'AutoFC',
         'n_enc': [256, 1],
         'n_dec': [1, None],
         'bias': True},
        {'type': 'Enc_Dec',  # 298
         'f_enc': [1, 5, 5, 5, 5, 5, 5],
         'n_enc': [256, 128, 64, 32, 16, 8, 4],
         'f_dec': [5, 5, 5, 5, 5, 5, 5],
         'n_dec': [4, 8, 16, 32, 64, 128, None],
         'f_conv': [5, 5, 1],
         'ups': gc.WEI,
         'downs': gc.WEI},
        # {'type': 'AutoConv',
        #  'f_enc': [1, 2, 3, 5],
        #  'kernel_enc': 10,
        #  'f_dec': [5, 3, 2, 1],
        #  'kernel_dec': 10},
        {'type': 'Enc_Dec',  # 132
         'f_enc': [1, 3, 5, 5, 5, 5],
         'n_enc': [256, 128, 64, 16, 8, 4],
         'f_dec': [5, 5, 5, 5, 3, 3],
         'n_dec': [4, 8, 16, 64, 128, None],
         'f_conv': [3, 3, 1],
         'ups': gc.WEI,
         'downs': gc.WEI},
        {'type': 'AutoConv',
         'convs': CONV1},
        {'type': 'Enc_Dec',  # HalfWeigths
         'f_enc': [1, 3, 3, 3, 3],
         'n_enc': [256, 64, 16, 8, 4],
         'f_dec': [3, 3, 3, 3, 3],
         'n_dec': [4, 8, 16, 64, None],
         'f_conv': [3, 3, 1],
         'ups': gc.WEI,
         'downs': gc.WEI},
        {'type': 'AutoConv',
         'convs': CONV2}]


N_EXPS = len(EXPS)


def run(id, Gs, signals, lrn, pert):
    Gx, Gy = ds.nodes_perturbated_graphs(Gs['params'], pert, seed=SEED)
    data = ds.LinearDS2GS(Gx, Gy, signals['samples'], signals['L'],
                          signals['deltas'], median=signals['median'],
                          same_coeffs=signals['same_coeffs'])
    data.to_unit_norm()
    data.add_noise(signals['noise'], test_only=signals['test_only'])
    data.to_tensor()

    epochs = 0
    mean_err = np.zeros(N_EXPS)
    med_err = np.zeros(N_EXPS)
    mse = np.zeros(N_EXPS)
    for i, exp in enumerate(EXPS):
        if exp['type'] == 'Linear':
            model = LinearModel(exp['N'])
        elif exp['type'] == 'Enc_Dec':
            exp['n_dec'][-1] = Gy.N
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
            conv = exp['convs'][PERT.index(pert)]
            net = ConvAutoencoder(conv['f_enc'], conv['kernel_enc'],
                                  conv['f_dec'], conv['kernel_dec'])
        elif exp['type'] == 'AutoFC':
            exp['n_dec'][-1] = Gy.N
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
    Gs['n_graphs'] = 1
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
    Gs['pert'] = PERT

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
    mean_err = np.zeros((Gs['n_graphs'], N_EXPS, len(PERT)))
    median_err = np.zeros((Gs['n_graphs'], N_EXPS, len(PERT)))
    mse = np.zeros((Gs['n_graphs'], N_EXPS, len(PERT)))
    for i, pert in enumerate(PERT):
        print('PCT:', pert)
        if Gs['n_graphs'] > 1:
            with Pool(processes=N_CPUS) as pool:
                results = []
                for j in range(Gs['n_graphs']):
                    results.append(pool.apply_async(run,
                                   args=[j, Gs, signals, learning, pert]))
                for j in range(Gs['n_graphs']):
                    mean_err[j, :, i], median_err[j, :, i], mse[j, :, i] = \
                        results[j].get()
        else:
            mean_err[0, :, i], median_err[0, :, i], mse[0, :, i] = \
                run(0, Gs, signals, learning, pert)

    # Print summary
        utils.print_partial_results(pert, EXPS, mean_err[:, :, i],
                                    median_err[:, :, i])
    end_time = time.time()
    utils.print_results(PERT, EXPS, mean_err, median_err)
    print('Time: {} hours'.format((end_time-start_time)/3600))

    if SAVE:
        utils.save(SAVE_PATH, EXPS, mean_err, median_err, Gs, signals,
                   learning)
