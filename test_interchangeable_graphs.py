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
VERBOSE = True
SAVE = True
SAVE_PATH = './results/perturbation'
EVAL_F = 1


EXPS = [{'type': 'Enc_Dec',
         'f_enc': [1, 5, 5, 10, 10, 15, 15],
         'n_enc': [256, 128, 64, 32, 16, 8, 4],
         'f_dec': [15, 15, 10, 10, 5, 5, 5],
         'n_dec': [4, 8, 16, 32, 64, 124, 256],
         'f_conv': [5, 5, 1],
         'ups': gc.WEI,
         'downs': gc.WEI},
        {'type': 'AutoConv',
         'f_enc': [1, 5, 5, 8],
         'kernel_enc': 10,
         'f_dec': [8, 5, 5, 1],
         'kernel_dec': 10},
        {'type': 'AutoFC',
         'n_enc': [256, 3],
         'n_dec': [3, 256],
         'bias': True}]

N_EXPS = len(EXPS)


def train_models(Gs, signals, lrn):
    Gx, Gy = ds.perturbated_graphs(Gs['params'], Gs['create'], Gs['destroy'],
                                   pct=Gs['pct'], seed=SEED)
    data = ds.LinearDS2GS(Gx, Gy, signals['samples'], signals['L'],
                          signals['deltas'], median=signals['median'],
                          same_coeffs=signals['same_coeffs'])
    data.to_unit_norm()
    data.add_noise(signals['noise'], test_only=signals['test_only'])
    sign_dist = np.median(np.linalg.norm(data.train_X-data.train_Y, axis=1))
    print('Distance signals:', sign_dist)
    data.to_tensor()

    epochs = np.zeros(N_EXPS)
    med_err = np.zeros(N_EXPS)
    mse = np.zeros(N_EXPS)
    models_state = []
    for i, exp in enumerate(EXPS):
        # Create model
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
        # Fit models
        epochs[i], _, _ = model.fit(data.train_X, data.train_Y, data.val_X,
                                    data.val_Y)
        _, med_err[i], mse[i] = model.test(data.test_X, data.test_Y)
        models_state.append(model.state_dict())
    

        

        # DEBUG!
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
            model = Model(net)
        model.load_state_dict(models_state[i])
        _, med_error, mse_error = model.test(data.test_X, data.test_Y)
        print('DEBUG: Original Graph {}-{} ({}): mse {} - MedianErr: {}'
              .format(i, exp['type'], model.count_params(),
                      mse_error, med_error))

    for i, exp in enumerate(EXPS):
        print('Original Graph: {}-{} ({}): epochs {} - mse {} - MedianErr: {}'
              .format(i, exp['type'], model.count_params(), epochs[i],
                      mse[i], med_err[i]))
    print()
    return models_state


def test_other_graphs(Gs, signals, lrn, models_state):
    med_err = np.zeros((Gs['n_graphs'], N_EXPS))
    mse = np.zeros((Gs['n_graphs'], N_EXPS))
    for i in Gs['n_graphs']:
        Gx, Gy = ds.perturbated_graphs(Gs['params'], Gs['create'], Gs['destroy'],
                                       pct=Gs['pct'], seed=SEED)
        data = ds.LinearDS2GS(Gx, Gy, signals['samples'], signals['L'],
                              signals['deltas'], median=signals['median'],
                              same_coeffs=signals['same_coeffs'])
        data.to_unit_norm()
        data.add_noise(signals['noise'], test_only=signals['test_only'])
        sign_dist = np.median(np.linalg.norm(data.train_X-data.train_Y, axis=1))
        print('Distance signals:', sign_dist)
        data.to_tensor()

        # Create models
        for j, exp in enumerate(EXPS):
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
                model = Model(net)
            model.load_state_dict(models_state[j])
            _, med_err[i, j], mse[i, j] = model.test(data.test_X, data.test_Y)

            # PRINT
    return med_err, mse


if __name__ == '__main__':
    # Set seeds
    np.random.seed(SEED)
    manual_seed(SEED)

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
    Gs['pct'] = True
    Gs['create'] = 10
    Gs['destroy'] = 10

    # Signals
    signals = {}
    signals['L'] = 6
    signals['samples'] = [5000, 1000, 1000]
    signals['deltas'] = k
    signals['noise'] = 0
    signals['median'] = True
    signals['same_coeffs'] = True
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
    models_state = train_models(Gs, signals, learning)
    # test_other_graphs(Gs, signals, learning, models_state)

    end_time = time.time()
    print('Time: {} hours'.format((end_time-start_time)/3600))
