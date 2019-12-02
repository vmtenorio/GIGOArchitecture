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
signals['pct'] = True
if signals['pct']:
    signals['eps1'] = 10
    signals['eps2'] = 10
else:
    signals['eps1'] = 0.1
    signals['eps2'] = 0.3

signals['median'] = True

# NN Parameters
nn_params = {}
nn_params['Fi'] = [1, int(N/32), int(N/16), N]
nn_params['Fo'] = [N, int(N/16), int(N/32)]
nn_params['Ki'] = 2
nn_params['Ko'] = 2
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

F_list = [[1, int(N/8), N],
          [1, N],
          [1, N],
          [1, int(N/64), int(N/32), int(N/16), N],
          [1, int(N/16), int(N/8), N]]

C_list = [[int(N/16), int(N/8), 1],
          [int(N/8), int(N/16), 1],
          [],
          [],
          [int(N/16), int(N/16), 1]]

# Testing K
param_list = [1, 2, 3, 4, 5]


def test_model(id, signals, nn_params, model_params):
    Gx, Gy = data_sets.perturbated_graphs(signals['g_params'], signals['eps1'], signals['eps2'],
                                          pct=signals['pct'], perm=signals['perm'])

    # Define the data model
    data = data_sets.LinearDS2GSLinksPert(Gx, Gy,
                                          signals['N_samples'],
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

    if not os.path.isfile('./out_hyp.csv'):
        outf = open('out_hyp.csv', 'w')
        outf.write('Experiment|Nodes|Communities|N samples|N graphs|' +
                   'Perturbation|L filter|Noise|' +
                   'F in|F out|K in|K out|C|' +
                   'Non Lin|Last Act Func|' +
                   'Batch size|Learning Rate|' +
                   'Num Params|MSE loss|Mean err|Median err|STD Median err|' +
                   'Mean t convergence|Mean epochs convergence\n')
    else:
        outf = open('out_hyp.csv', 'a')
    outf.write("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n".format(
            "LinksPert", N, k, signals['N_samples'], signals['N_graphs'],
            signals['eps1'], signals['L_filter'], signals['noise'],
            nn_params['Fi'], nn_params['Fo'], nn_params['Ki'], nn_params['Ko'], nn_params['C'],
            nn_params['nonlin'], nn_params['last_act_fn'],
            model_params['batch_size'], model_params['learning_rate'],
            n_params, mse_loss, mean_err, median_err, std_err,
            mean_t_conv, mean_ep_conv))
    outf.close()

    return np.median(med_errs)        # Keeping the one with the best median error


def get_Fout(fi, c0):
    fi[0] = c0
    fi.reverse()
    return fi

if __name__ == "__main__":

    best_err = 100000
    best_param = 0
    for i in range(2, len(F_list)):
        # Prepare here the parameter you want to test
        nn_params['Fi'] = F_list[i]
        if len(C_list[i]) > 0:
            nn_params['Fo'] = get_Fout(F_list[i].copy(), C_list[i][0])
        else:
            nn_params['Fo'] = get_Fout(F_list[i].copy(), 1)
        nn_params['C'] = C_list[i]
        # Until here
        err = test_arch(signals, nn_params, model_params)
        if err < best_err:
            best_err = err
            best_param = i
    print("Best Parameter: F - {}, C - {}".format(F_list[best_param], C_list[best_param]))


#############################################
# Test n2 paper: add noise to the signal

# To line 92
# param_list = [0, .025, .05, 0.75, .1]

# To line 194
# signals['noise'] = p


#############################################
# Test n3 paper: perturbate links
# To line 92
# param_list = [5, 10, 15, 20]

# To line 194
# signals['eps1'] = p
# signals['eps2'] = p