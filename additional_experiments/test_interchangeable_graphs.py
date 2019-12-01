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
VERBOSE = False
SAVE = False
SAVE_PATH = './results/diff_graphs'
EVAL_F = 1


EXPS = [
        # {'type': 'Enc_Dec',  # 132
        #  'f_enc': [1, 3, 3, 3, 3],
        #  'n_enc': [64]*5,
        #  'f_dec': [3, 3, 3, 3, 3],
        #  'n_dec': [64]*5,
        #  'f_conv': [3, 3, 1],
        #  'ups': None,
        #  'downs': None},
        {'type': 'Enc_Dec',  # 132
         'f_enc': [1, 3, 3, 3, 3],
         'n_enc': [64, 32, 16, 8, 4],
         'f_dec': [3, 3, 3, 3, 3],
         'n_dec': [4, 8, 16, 32, 54],
         'f_conv': [3, 3, 1],
         'ups': gc.WEI,
         'downs': gc.WEI},
        {'type': 'AutoConv',  # 140
         'f_enc': [1, 2, 2, 2, 2],
         'kernel_enc': 5,
         'f_dec': [2, 2, 2, 2, 1],
         'kernel_dec': 5},
        {'type': 'AutoFC',  # 128
         'n_enc': [64, 1],
         'n_dec': [1, 64],
         'bias': True}]

N_EXPS = len(EXPS)


def create_model(Gx, Gy, exp, lrn):
    if exp['type'] == 'Linear':
        model = LinearModel(exp['N'])
    elif exp['type'] == 'Enc_Dec':
        clust_x = gc.MultiResGraphClustering(Gx, exp['n_enc'],
                                             k=exp['n_enc'][-1],
                                             up_method=exp['downs'])
        clust_y = gc.MultiResGraphClustering(Gy, exp['n_dec'],
                                             k=exp['n_enc'][-1],
                                             up_method=exp['ups'])

        clust_x.plot_labels(False)
        clust_y.plot_labels()
        sys.exit()

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
    return model


def train_models(Gs, signals, lrn):
    # Create data
    # Gx, Gy = ds.perturbated_graphs(Gs['params'], Gs['create'], Gs['destroy'],
    #                                pct=Gs['pct'], seed=SEED)
    Gx, Gy = ds.nodes_perturbated_graphs(Gs['params'], 10, seed=SEED)
    data = ds.LinearDS2GSNodesPert(Gx, Gy, signals['samples'], signals['L'],
                                   signals['deltas'], median=signals['median'],
                                   same_coeffs=signals['same_coeffs'])

    data.to_unit_norm()
    data.add_noise(signals['noise'], test_only=signals['test_only'])

    data.to_tensor()
    data_state = data.state_dict()

    med_err = np.zeros(N_EXPS)
    epochs = np.zeros(N_EXPS)
    mse = np.zeros(N_EXPS)
    models_states = []
    for i, exp in enumerate(EXPS):
        model = create_model(Gx, Gy, exp, lrn)
        # Fit models
        epochs[i], _, _ = model.fit(data.train_X, data.train_Y, data.val_X,
                                    data.val_Y)
        _, med_error, mse_error = model.test(data.test_X, data.test_Y)
        models_states.append(model.state_dict())
        print('Original Graph {}-{} ({}): mse {} - MedianErr: {}'
              .format(i, exp['type'], model.count_params(),
                      mse_error, med_error))
    print()
    return data_state, models_states, Gx, Gy


def test_original_graphs(data_state, models_state, Gx, Gy, signals, lrn):
    data = ds.LinearDS2GS(Gx, Gy, signals['samples'], signals['L'],
                          signals['deltas'], median=signals['median'],
                          same_coeffs=signals['same_coeffs'])
    data.load_state_dict(data_state, unit_norm=True)
    data.add_noise(signals['noise'], test_only=signals['test_only'])
    sign_dist = np.median(np.linalg.norm(data.train_X-data.train_Y, axis=1))
    print('Distance signals:', sign_dist)
    data.to_tensor()
    for i, exp in enumerate(EXPS):
        model = create_model(Gx, Gy, exp, lrn)
        model.load_state_dict(models_state[i])
        _, med_error_train, _ = model.test(data.train_X, data.train_Y)
        _, med_error_test, _ = model.test(data.test_X, data.test_Y)
        print('Original (debug) Graph {}-{} ({}): TrainErr {} - TestErr: {}'
              .format(i, exp['type'], model.count_params(),
                      med_error_train, med_error_test))
    print()


def test_other_graphs(Gs, signals, lrn, data_state, models_state):
    med_err = np.zeros((Gs['n_graphs'], N_EXPS))
    mse = np.zeros((Gs['n_graphs'], N_EXPS))
    for i in range(Gs['n_graphs']):
        Gx, Gy = ds.perturbated_graphs(Gs['params'], Gs['create'], Gs['destroy'],
                                       pct=Gs['pct'], seed=SEED)
        data = ds.LinearDS2GS(Gx, Gy, signals['samples'], signals['L'],
                              signals['deltas'], median=signals['median'],
                              same_coeffs=signals['same_coeffs'])
        data.load_state_dict(data_state, unit_norm=True)
        data.add_noise(signals['noise'], test_only=signals['test_only'])
        sign_dist = np.median(np.linalg.norm(data.train_X-data.train_Y,
                              axis=1))
        print('Distance signals:', sign_dist)
        data.to_tensor()

        # Create models
        for j, exp in enumerate(EXPS):
            model = create_model(Gx, Gy, exp, lrn)
            model.load_state_dict(models_state[j])
            _, med_err[i, j], mse[i, j] = model.test(data.test_X, data.test_Y)
            print('Graph {}: {}-{} ({}): mse {} - MedianErr: {}'
                  .format(i, j, exp['type'], model.count_params(),
                          mse[i, j], med_err[i, j]))
    return med_err, mse


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
    Gs['create'] = 10
    Gs['destroy'] = 10

    # Signals
    signals = {}
    signals['L'] = 6
    signals['samples'] = [2000, 500, 500]
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
    data_state, models_state, Gx, Gy = train_models(Gs, signals, learning)
    test_original_graphs(data_state, models_state, Gx, Gy, signals, learning)
    med_err, mse = test_other_graphs(Gs, signals, learning, data_state,
                                     models_state)
    utils.print_partial_results(signals['noise'], EXPS, mse, med_err)
    end_time = time.time()
    print('Time: {} hours'.format((end_time-start_time)/3600))
