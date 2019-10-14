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
PCT = [[5, 5]]


EXPS = [{'type': 'Linear', 'N': 256},
        {'type': 'Enc_Dec',
         'f_enc': [1, 5, 5, 10, 10, 15, 15],
         'n_enc': [256, 128, 64, 32, 16, 8, 4],
         'f_dec': [15, 15, 10, 10, 5, 5, 5],
         'n_dec': [4, 8, 16, 32, 64, 124, 256],
         'f_conv': [5, 5, 1],
         'ups': gc.WEI}, ]
        # {'type': 'Enc_Dec',
        #  'f_enc': [1, 5, 10, 15],
        #  'n_enc': [256, 64, 16, 4],
        #  'f_dec': [15, 10, 10, 10],
        #  'n_dec': [4, 16, 64, 256],
        #  'f_conv': [10, 5, 1],
        #  'ups': gc.WEI},
        # {'type': 'Enc_Dec',
        #  'f_enc': [1, 10, 10, 20, 20, 30, 30],
        #  'n_enc': [256, 128, 64, 32, 16, 8, 4],
        #  'f_dec': [30, 30, 20, 20, 10, 10, 10],
        #  'n_dec': [4, 8, 16, 32, 64, 124, 256],
        #  'f_conv': [10, 10, 1],
        #  'ups': gc.WEI}]

N_EXPS = len(EXPS)


def test_models(id, Gs, signals, lrn, pct):
    Gx, Gy = ds.perturbated_graphs(Gs['params'], pct[0], pct[1], pct=True,
                                   seed=SEED)
    print('Links different(%):', np.sum(Gx.A != Gy.A)/2/Gx.Ne*100,
          'of ', Gx.Ne)

    # TODO: print signal distance
    data = ds.LinearDS2GS(Gx, Gy, signals['samples'], signals['L'],
                          signals['deltas'], median=signals['median'],
                          same_coeffs=signals['same_coeffs'])

    # data = ds.NonLinearDS2GS(Gx, Gy, signals['samples'], signals['L'],
    #                          signals['deltas'], median=signals['median'],
    #                          same_coeffs=signals['same_coeffs'])

    data.to_unit_norm()
    data.add_noise(signals['noise'], test_only=True)
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
                                      As_enc=clust_x.As, act_fn=lrn['af'],
                                      last_act_fn=lrn['laf'], ups=exp['ups'],
                                      downs=exp['ups'])
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
    G_params['type'] = ds.SBM  # SBM or ER
    G_params['N'] = N = 256
    G_params['k'] = k = 4
    G_params['p'] = 0.2
    G_params['q'] = 0.015/4
    G_params['type_z'] = ds.RAND
    Gs['params'] = G_params

    # Signals
    signals = {}
    signals['L'] = 5
    signals['samples'] = [5000, 1000, 1000]
    signals['deltas'] = k
    signals['noise'] = 0
    signals['median'] = True
    signals['same_coeffs'] = True

    learning = {}
    learning['laf'] = nn.Tanh()
    learning['af'] = nn.Tanh()
    learning['lr'] = 0.1
    learning['dr'] = 0.9
    learning['batch'] = 100
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
                    results.append(pool.apply_async(test_models,
                                   args=[j, Gs, signals, learning, pct]))
                for j in range(Gs['n_graphs']):
                    mean_err[j, :, i], median_err[j, :, i], mse[j, :, i] = \
                        results[j].get()
        else:
            mean_err[0, :, i], median_err[0, :, i], mse[0, :, i] = \
                test_models(0, Gs, signals, learning, pct)

    # Print summary
        utils.print_partial_results(pct, EXPS, mean_err[:, :, i],
                                    median_err[:, :, i])

    end_time = time.time()
    utils.print_results(PCT, EXPS, mean_err, median_err)
    print('Time: {} hours'.format((end_time-start_time)/3600))

    if SAVE:
        utils.save(SAVE_PATH, EXPS, mean_err, median_err, Gs, signals,
                   learning)
