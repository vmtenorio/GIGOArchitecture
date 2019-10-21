import os
import torch
import torch.nn as nn
import numpy as np

from cnngs_src import graphtools, datatools

from GIGO.model import Model

from graph_enc_dec import data_sets
from GIGO.arch import GIGOArch

from multiprocessing import Pool, cpu_count

from time import time

TB_LOG = False
VERB = False
ARCH_INFO = False
N_CPUS = cpu_count()

# Parameters

# Data parameters
N_samples = 5000
eval_freq = 4
N_graphs = 1

L_filter = 6

num_epochs = 40
batch_size = 100
max_non_dec = 5

# Graph parameters
N = 64
k = 4
G_params = {}
G_params['type'] = data_sets.SBM #SBM or ER
G_params['N'] = N
G_params['k'] = k
G_params['p'] = [0.6, 0.7, 0.6, 0.8]
G_params['q'] = 0.2
G_params['type_z'] = data_sets.RAND
pct = True
if pct:
    eps1 = 5
    eps2 = 5
else:
    eps1 = 0.1
    eps2 = 0.3
median = True

loss_func = nn.MSELoss()

# NN Parameters
K = 2
Fi = [1,int(N/2),N]
Fo = [N,int(N/2),int(N/4)]
C = [Fo[-1],int(N/4),1]
nonlin_s = "relu"
if nonlin_s == "relu":
    nonlin = nn.ReLU
elif nonlin_s == "tanh":
    nonlin = nn.Tanh
elif nonlin_s == "sigmoid":
    nonlin = nn.Sigmoid
batch_norm = True

optimizer = "ADAM"
beta1 = 0.9
beta2 = 0.999
decay_rate = 0.99

model_param = {}

model_param['optimizer'] = optimizer
model_param['beta1'] = beta1
model_param['beta2'] = beta2
model_param['decay_rate'] = decay_rate
model_param['loss_func'] = loss_func
model_param['num_epochs'] = num_epochs
model_param['eval_freq'] = eval_freq
model_param['max_non_dec'] = max_non_dec
model_param['tb_log'] = TB_LOG
model_param['verb'] = VERB

# Hyperparameters tuning

learning_rate_list = [0.05,0.01,0.1]

F_list = [[1, int(N/2), N],
    [1, N],
    [1, int(N/4), int(N/2), N],
    [1, int(N/4), N]]

K_list = [2,3,4]

batch_size_list = [100, 200]

C_list = [[int(N/2),int(N/4),1],
    [int(N/2),1],
    [int(N/2),int(N/2),int(N/4),1]]

nonlin_list = [nn.Tanh, nn.Sigmoid, nn.ReLU]

batch_norm_list = [True, False]

def run_arch(model_param, Fi, Fo, Ki, Ko, C, nl, bn):
    global G_params
    Gx, Gy = data_sets.perturbated_graphs(G_params, eps1, eps2)

    # Define the data model
    data = data_sets.LinearDS2GS(Gx, Gy, N_samples, L_filter, G_params['k'], median=median)
    data.to_unit_norm()

    Gx.compute_laplacian('normalized')
    Gy.compute_laplacian('normalized')

    archit = GIGOArch(Gx.L.todense(), Gy.L.todense(), Fi, Fo, Ki, Ko, C, nl, bn, ARCH_INFO)

    model_param['arch'] = archit

    model = Model(**model_param)
    mse_loss, _, mean_norm_err = model.eval(data.train_X, data.train_Y, data.val_X, data.val_Y, data.test_X, data.test_Y)

    return mse_loss, mean_norm_err, archit.n_params, model.t_conv, model.epochs_conv

def test_arch(lr, k, bs, fi, fo, c, nl, bn):

    model_param['learning_rate'] = lr
    model_param['batch_size'] = bs
    print("Testing: F = {}, K = {}, C = {}, BS = {}, LR = {}, Nonlin = {}".format(\
            fi, k, c, bs, lr, nl))

    pool = Pool(processes=N_CPUS)

    if nl == "relu":
        nonlin = nn.ReLU
    elif nl == "tanh":
        nonlin = nn.Tanh
    elif nl == "sigmoid":
        nonlin = nn.Sigmoid

    results = []
    for ng in range(N_graphs):
        results.append(pool.apply_async(run_arch,\
                        args=[model_param, fi, fo, k, k, c, nonlin, bn]))

    mean_norm_errs = np.zeros(N_graphs)
    mse_losses = np.zeros(N_graphs)
    t_conv = np.zeros(N_graphs)
    epochs_conv = np.zeros(N_graphs)
    for ng in range(N_graphs):
        # No problem in overriding n_params, as it has always the same value
        mse_losses[ng], mean_norm_errs[ng], n_params, t_conv[ng], epochs_conv[ng] = results[ng].get()

    mse_loss = np.mean(mse_losses)
    mean_norm_err = np.mean(mean_norm_errs)
    median_mean_norm_err = np.median(mean_norm_errs)
    std_mean_norm_err = np.std(mean_norm_errs)
    mean_t_conv = round(np.mean(t_conv), 6)
    mean_ep_conv = np.mean(epochs_conv)

    print("--------------------------------------Ended simulation--------------------------------------")
    print("2G difussed deltas architecture parameters")
    print("Graph: N = {}; k = {}".format(str(N), str(k)))
    print("Fin: {}, Fout: {}, Kin: {}, Kout: {}, C: {}".format(fi, fo, k, k, c))
    print("Non lin: " + str(nl))
    print("N params: " + str(n_params))
    #print("MSE loss = {}".format(str(mse_losses)))
    print("MSE loss mean = {}".format(mse_loss))
    #print("Mean Squared Error = {}".format(str(mean_norm_errs)))
    print("Mean Norm Error = {}".format(mean_norm_err))
    print("Median error = {}".format(median_mean_norm_err))
    print("STD = {}".format(std_mean_norm_err))
    print("Until convergence: Time = {} - Epochs = {}".format(mean_t_conv, mean_ep_conv))

    if not os.path.isfile('./out_hyp.csv'):
        out = open('out_hyp.csv', 'w')
        out.write('Nodes|Communities|N samples|Batch size|' +
                    'F in|F out|K in|K out|C|Learning Rate|Non Lin|Median|Batch norm|' +
                    'MSE loss|Mean norm err|Median mean norm err|STD mean norm err|' +
                    'Mean t convergence|Mean epochs convergence\n')
    else:
        out = open('out_hyp.csv', 'a')
    out.write("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n".format(
                    N, k, N_samples, batch_size,
                    fi, fo, ki, ko, c, lr, nl, median, batch_norm,
                    mse_loss, mean_norm_err, median_mean_norm_err, std_mean_norm_err,
                    mean_t_conv, mean_ep_conv))
    out.close()

    return np.median(mean_norm_errs)        # Keeping the one with the best median error

def check_err(param, old_param, err, best_err):
    if err < best_err:
        return param
    else:
        return old_param


if __name__ == "__main__":
    best_err = 1000000
    for lr in learning_rate_list:
        err = test_arch(lr, K, batch_size, Fi, Fo, C, nonlin_s, batch_norm)
        learning_rate, best_err = check_err(lr, learning_rate, err, best_err)
    best_err = 1000000
    for k in K_list:
        err = test_arch(learning_rate, k, batch_size, Fi, Fo, C, nonlin_s, batch_norm)
        K, best_err = check_err(k, K, err, best_err)
    best_err = 1000000
    for bs in batch_size_list:
        err = test_arch(learning_rate, K, bs, Fi, Fo, C, nonlin_s, batch_norm)
        batch_size, best_err = check_err(bs, batch_size, err, best_err)
    best_err = 1000000
    for f in F_list:
        fo = f.copy()
        fo[0] = int(N/2)
        fo.reverse()
        fi = f
        err = test_arch(learning_rate, K, batch_size, fi, fo, C, nonlin_s, batch_norm)
        Fi, best_err = check_err(fi, Fi, err, best_err)
        Fo = Fi.copy()
        Fo[0] = int(N/2)
        Fo.reverse()
    best_err = 1000000
    for c in C_list:
        err = test_arch(learning_rate, K, batch_size, Fi, Fo, c, nonlin_s, batch_norm)
        C, best_err = check_err(c, C, err, best_err)
    best_err = 1000000
    for nl in nonlin_list:
        err = test_arch(learning_rate, K, batch_size, Fi, Fo, C, nl, batch_norm)
        nonlin_s, best_err = check_err(nl, nonlin_s, err, best_err)
    best_err = 1000000
    for bn in batch_norm_list:
        err = test_arch(learning_rate, K, batch_size, Fi, Fo, C, nonlin_s, bn)
        batch_norm, best_err = check_err(bn, batch_norm, err, best_err)
