import os
import time
import torch
import torch.nn as nn
import numpy as np

from graph_enc_dec.model import Model, ADAM

from graph_enc_dec import data_sets
from GIGO.arch import GIGOArch

from multiprocessing import Pool, cpu_count

TB_LOG = False
VERB = False
ARCH_INFO = False
N_CPUS = cpu_count()

# Parameters

# Data parameters
signals = {}
signals['N_samples'] = 2000
signals['N_graphs'] = 16
signals['L_filter'] = 6
signals['noise'] = 0
signals['test_only'] = True

# Graph parameters
G_params = {}
G_params['type'] = data_sets.SBM
G_params['N'] = N = 128
G_params['k'] = k = 4
G_params['p'] = 0.3
G_params['q'] = [[0, 0.0075, 0, 0.0],
                 [0.0075, 0, 0.004, 0.0025],
                 [0, 0.004, 0, 0.005],
                 [0, 0.0025, 0.005, 0]]
G_params['type_z'] = data_sets.RAND
signals['g_params'] = G_params

signals['perm'] = True

signals['median'] = True
signals['pert'] = 32
Nout = N - signals['pert']

# NN Parameters
nn_params = {}
nn_params['Fi'] = [1, int(Nout/2), Nout]
nn_params['Fo'] = [N, int(N/16), int(N/32), int(N/64), 1]
nn_params['Ki'] = 3
nn_params['Ko'] = 3
nn_params['C'] = [nn_params['Fo'][-1], int(N/16), 1]
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
model_params['learning_rate'] = 0.05
model_params['decay_rate'] = 0.99
model_params['loss_func'] = nn.MSELoss()
model_params['epochs'] = 200
model_params['batch_size'] = 50
model_params['eval_freq'] = 4
model_params['max_non_dec'] = 10
model_params['verbose'] = VERB

# Hyperparameters tuning

F_list = [[1, 12, Nout],
          [1, Nout],
          [1, Nout],
          [1, 2, 4, 8, Nout],
          [1, 2, 4, 8, Nout]]

Fo_list = [[N, 16, int(N/16)],
           [N, 1],
           [N, 1],
           [N, 8, 4, 2, 1],
           [N, 8, 4, 2, 2]]

C_list = [[int(N/16), int(N/8), 1],
          [int(N/8), int(N/16), 1],
          [],
          [],
          [2, 8, 1]]

# F_list = [[1, int(N/8), N],
#           [1, N],
#           [1, int(N/64), int(N/32), int(N/16), N]]

# Testing K
param_list = [10, 20, 30, 40, 50]


def test_model(id, signals, nn_params, model_params):
    Gx, Gy = data_sets.nodes_perturbated_graphs(signals['g_params'], signals['pert'],
                                                perm=signals['perm'])

    # Define the data model
    data = data_sets.LinearDS2GSNodesPert(Gx, Gy, signals['N_samples'],
                                          signals['L_filter'], signals['g_params']['k'],  # k is n_delts
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

    print("DONE {}: MSE={} - Mean Err={} - Median Err={} - Params={} - t_conv={} - epochs={}".format(
        id, mse, mean_err, med_err, model.count_params(), round(t_conv, 4), epochs
    ))

    return mse, mean_err, med_err, model.count_params(), t_conv, epochs


def test_arch(signals, nn_params, model_params):

    print("Testing: F = {}, K = {}, C = {}, BS = {}, LR = {}, Nonlin = {}".format(\
          nn_params['Fi'], nn_params['Ki'], nn_params['C'],
          model_params['batch_size'], model_params['learning_rate'], nn_params['nonlin']))

    pool = Pool(processes=N_CPUS)

    results = []
    for ng in range(signals['N_graphs']):
        results.append(pool.apply_async(test_model,
                        args=[ng, signals, nn_params, model_params]))

    mean_errs = np.zeros(signals['N_graphs'])
    mse_losses = np.zeros(signals['N_graphs'])
    med_errs = np.zeros(signals['N_graphs'])
    t_conv = np.zeros(signals['N_graphs'])
    epochs_conv = np.zeros(signals['N_graphs'])
    for ng in range(signals['N_graphs']):
        # No problem in overriding n_params, as it has always the same value
        mse_losses[ng], mean_errs[ng], med_errs[ng], n_params, t_conv[ng], epochs_conv[ng] = results[ng].get()

    mse_loss = np.median(mse_losses)
    mean_err = np.median(mean_errs)
    median_err = np.median(med_errs)
    std_err = np.std(med_errs)
    mean_t_conv = round(np.mean(t_conv), 6)
    mean_ep_conv = np.mean(epochs_conv)
    print("-----------------------------------------------------------------------------------")
    print("DONE Test: MSE={} - Mean Err={} - Median Err={} - Params={} - t_conv={} - epochs={}".format(
        mse_loss, mean_err, median_err, n_params, mean_t_conv, mean_ep_conv
    ))
    print("-----------------------------------------------------------------------------------")

    if not os.path.isfile('./out_exps.csv'):
        outf = open('out_hyp.csv', 'w')
        outf.write('Experiment|Nodes|Communities|N samples|N graphs|' +
                   'Perturbation|L filter|Noise|' +
                   'F in|F out|K in|K out|C|' +
                   'Non Lin|Last Act Func|' +
                   'Batch size|Learning Rate|' +
                   'Num Params|MSE loss|Mean err|Median err|STD Median err|' +
                   'Mean t convergence|Mean epochs convergence\n')
    else:
        outf = open('out_exps.csv', 'a')
    outf.write("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n".format(
            "NodesPert", N, k, signals['N_samples'], signals['N_graphs'],
            signals['pert'], signals['L_filter'], signals['noise'],
            nn_params['Fi'], nn_params['Fo'], nn_params['Ki'], nn_params['Ko'], nn_params['C'],
            nn_params['nonlin'], nn_params['last_act_fn'],
            model_params['batch_size'], model_params['learning_rate'],
            n_params, mse_loss, mean_err, median_err, std_err,
            mean_t_conv, mean_ep_conv))
    outf.close()

    return np.median(med_errs)        # Keeping the one with the best median error


if __name__ == "__main__":

    best_err = 100000
    best_param = 0
    for p in range(len(F_list)):
        # Prepare here the parameter you want to test
        # signals['pert'] = p
        # Nout = N - p
        nn_params['Fi'] = F_list[p]
        nn_params['Fo'] = Fo_list[p]
        nn_params['C'] = C_list[p]
        # Until here
        err = test_arch(signals, nn_params, model_params)
        if err < best_err:
            best_err = err
            best_param = p
    print("Best Parameter: {}".format(best_param))


#############################################
# Test n1 paper: Delete nodes

# To line 87
# param_list = [10, 20, 30, 40, 50]

# To line 188
# signals['pert'] = p
# Nout = N - p
# nn_params['Fi'] = [1, int(Nout/2), Nout]

#############################################
# Test n2 paper: add noise to the signal

# To line 87
# param_list = [0, .025, .05, 0.75, .1]

# To line 188
# signals['noise'] = p