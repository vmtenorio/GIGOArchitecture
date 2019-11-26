import os
import torch.nn as nn
import numpy as np
import time

from numpy.core._multiarray_umath import ndarray

from graph_enc_dec.model import Model, ADAM
from graph_enc_dec import data_sets
from GIGO.arch import GIGOArch

from multiprocessing import Pool, cpu_count

SEED = None
TB_LOG = False
VERB = False
ARCH_INFO = True
N_CPUS = cpu_count()

# Parameters

# Data parameters
signals = {}
signals['N_samples'] = 2000
signals['N_graphs'] = 15
signals['L_filter'] = 6
signals['noise'] = 0
signals['test_only'] = True

# Graph parameters
N = 128
k = 4
G_params = {}
G_params['type'] = data_sets.SBM
G_params['N'] = N
G_params['k'] = k
G_params['p'] = [0.6, 0.7, 0.6, 0.8]
# With 64 nodes it doesn't get connected like this
# G_params['q'] = [[0, 0.0075, 0, 0.0],
#                  [0.0075, 0, 0.004, 0.0025],
#                  [0, 0.004, 0, 0.005],
#                  [0, 0.0025, 0.005, 0]]
G_params['q'] = 0.2
G_params['type_z'] = data_sets.RAND
signals['g_params'] = G_params

signals['perm'] = True
signals['pert'] = 30

signals['median'] = True

# NN Parameters
nn_params = {}
nn_params['Fi'] = [1, N - signals['pert']]
nn_params['Fo'] = [N, 1]
nn_params['Ki'] = 3
nn_params['Ko'] = 3
nn_params['C'] = []
nonlin_s = "tanh"
if nonlin_s == "relu":
    nn_params['nonlin'] = nn.ReLU
elif nonlin_s == "tanh":
    nn_params['nonlin'] = nn.Tanh
elif nonlin_s == "sigmoid":
    nn_params['nonlin'] = nn.Sigmoid
else:
    nn_params['nonlin'] = None
nn_params['last_act_fn'] = nn.Tanh
nn_params['batch_norm'] = False
nn_params['arch_info'] = ARCH_INFO

# Model parameters
model_params = {}
model_params['opt'] = ADAM
model_params['learning_rate'] = 0.01
model_params['decay_rate'] = 0.99
model_params['loss_func'] = nn.MSELoss()
model_params['epochs'] = 100
model_params['batch_size'] = 50
model_params['eval_freq'] = 4
model_params['max_non_dec'] = 10
model_params['verbose'] = VERB


def test_model(signals, nn_params, model_params):
    Gx, Gy = data_sets.nodes_perturbated_graphs(signals['g_params'], signals['pert'],
                                                perm=signals['perm'], seed=SEED)

    # Define the data model
    data = data_sets.LinearDS2GSNodesPert(Gx, Gy, signals['N_samples'],
                                          signals['L_filter'], signals['g_params']['k'],    # k is n_delts
                                          median=signals['median'])
    data.to_unit_norm()
    data.add_noise(signals['noise'], test_only=signals['test_only'])
    data.to_tensor()

    Gx.compute_laplacian('normalized')
    Gy.compute_laplacian('normalized')

    archit = GIGOArch(Gx.L.todense(), Gy.L.todense(),
                      nn_params['Fi'], nn_params['Fo'], nn_params['Ki'], nn_params['Ko'], nn_params['C'],
                      nn_params['nonlin'], nn_params['last_act_fn'], nn_params['batch_norm'],
                      nn_params['arch_info'])

    model_params['arch'] = archit

    model = Model(**model_params)
    t_init = time.time()
    epochs, _, _ = model.fit(data.train_X, data.train_Y, data.val_X, data.val_Y)
    t_conv = time.time() - t_init
    mean_err, med_err, mse = model.test(data.test_X, data.test_Y)

    print("DONE: MSE={} - Mean Err={} - Median Err={} - Params={} - t_conv={} - epochs={}".format(
        mse, mean_err, med_err, model.count_params(), round(t_conv, 4), epochs
    ))

    return mse, med_err, mean_err, model.count_params(), t_conv, epochs


if __name__ == '__main__':
    pool = Pool(processes=N_CPUS)
    results = []
    for ng in range(signals['N_graphs']):
        print("Started test " + str(ng))
        results.append(pool.apply_async(test_model,
                                        args=[signals, nn_params, model_params]))

    mse_losses = np.zeros(signals['N_graphs'])
    mean_errs = np.zeros(signals['N_graphs'])
    med_errs = np.zeros(signals['N_graphs'])
    t_conv = np.zeros(signals['N_graphs'])
    epochs_conv: ndarray = np.zeros(signals['N_graphs'])
    for ng in range(signals['N_graphs']):
        # No problem in overriding n_params, as it has always the same value
        mse_losses[ng], med_errs[ng], mean_errs[ng], n_params, t_conv[ng], epochs_conv[ng] = results[ng].get()

    mse_loss = np.median(mse_losses)
    mean_err = np.mean(mean_errs)
    median_err = np.median(med_errs)
    std_err = np.std(med_errs)
    mean_t_conv = round(np.mean(t_conv), 6)
    mean_ep_conv = np.mean(epochs_conv)
    print("--------------------------------------Ended simulation--------------------------------------")
    print("2G difussed deltas architecture parameters")
    print("Graph: N = {}; k = {}".format(str(N), str(k)))
    print("Fin: {}, Fout: {}, Kin: {}, Kout: {}, C: {}".format(
        nn_params['Fi'], nn_params['Fo'], nn_params['Ki'], nn_params['Ko'], nn_params['C']))
    print("Non lin: " + str(nonlin_s))
    print("N params: " + str(n_params))
    # print("MSE loss = {}".format(str(mse_losses)))
    print("MSE loss mean = {}".format(mse_loss))
    # print("Mean Squared Error = {}".format(str(mean_norm_errs)))
    print("Mean Norm Error = {}".format(mean_err))
    print("Median error = {}".format(median_err))
    print("STD = {}".format(std_err))
    print("Until convergence: Time = {} - Epochs = {}".format(mean_t_conv, mean_ep_conv))
    if not os.path.isfile('./out.csv'):
        f = open('out.csv', 'w')
        f.write('Experiment|Nodes|Communities|N samples|N graphs|' +
                'Median|L filter|Noise|' +
                'F in|F out|K in|K out|C|' +
                'Non Lin|Last Act Func|' +
                'Batch size|Learning Rate|' +
                'MSE loss|Mean err|Mean err|STD Median err|' +
                'Mean t convergence|Mean epochs convergence\n')
    else:
        f = open('out.csv', 'a')
    f.write("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n".format(
        "NodesPert", N, k, signals['N_samples'], signals['N_graphs'],
        signals['median'], signals['L_filter'], signals['noise'],
        nn_params['Fi'], nn_params['Fo'], nn_params['Ki'], nn_params['Ko'], nn_params['C'],
        nonlin_s, nn_params['last_act_fn'],
        model_params['batch_size'], model_params['learning_rate'],
        mse_loss, mean_err, median_err, std_err,
        mean_t_conv, mean_ep_conv))
    f.close()
