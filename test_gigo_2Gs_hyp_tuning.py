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
VERB = True
ARCH_INFO = True
N_CPUS = cpu_count()

# Parameters

# Data parameters
N_samples = 5000
eval_freq = 4
N_graphs = 10

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

optimizer = "ADAM"
beta1 = 0.9
beta2 = 0.999
decay_rate = 0.99
nonlin = nn.Tanh

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

learning_rate = [0.05]

F = [[1, int(N/2), N],
    [1, N],
    [1, int(N/4), int(N/2), N],
    [1, int(N/4), N]]

K = [2,3,4]

batch_size = [100, 200]

C = [[int(N/2),int(N/4),1],
    []]

median = [True, False]

def run_arch(G_params, model_param, median, Fi, Fo, Ki, Ko):
    Gx, Gy = data_sets.perturbated_graphs(G_params, eps1, eps2)

    # Define the data model
    data = data_sets.LinearDS2GS(Gx, Gy, N_samples, L_filter, G_params['k'], median=median)
    data.to_unit_norm()

    Gx.compute_laplacian('normalized')
    Gy.compute_laplacian('normalized')

    archit = GIGOArch(Gx.L.todense(), Gy.L.todense(), Fi, Fo, Ki, Ko, nonlin)

    model_param['arch'] = archit

    model = Model(**model_param)
    mse_loss, _, mean_norm_err = model.eval(data.train_X, data.train_Y, data.val_X, data.val_Y, data.test_X, data.test_Y)

    return mse_loss, mean_norm_err, archit.n_params, model.t_conv, model.epochs_conv

def test_arch(G_params, model_param, median, Fi, Fo, Ki, Ko):

    G_params['N'] = n
    G_params['k'] = comm

    model_param['learning_rate'] = lr
    model_param['batch_size'] = bs
    print("Testing: N = {}, c = {}, F = {}, K = {}, BS = {}, LR = {}, Median = {}".format(\
            n, comm, f, k, bs, lr, m))

    t_init = time()
    mse_loss, mean_norm_err, median_mean_norm_err, std_mean_norm_err = test_arch(G_params, model_param, m, f, fo, k, k)
    t_spent = time() - t_init

    pool = Pool(processes=N_CPUS)

    results = []
    for ng in range(N_graphs):
        results.append(pool.apply_async(run_arch,\
                        args=[G_params, model_param, median, Fi, Fo, Ki, Ko]))

    mean_norm_errs = np.zeros(N_graphs)
    mse_losses = np.zeros(N_graphs)
    for ng in range(N_graphs):
        mse_losses[ng], mean_norm_errs[ng]  = results[ng].get()

    mse_loss = np.mean(mse_losses)
    mean_norm_err = np.mean(mean_norm_errs)
    median_mean_norm_err = np.median(mean_norm_errs)
    std_mean_norm_err = np.std(mean_norm_errs)
    mean_t_conv = round(np.mean(t_conv), 6)
    mean_ep_conv = np.mean(epochs_conv)

    print("Results: MSE loss = {}, Mean Norm Err = {}, Median Norm Err = {}, STD Norm Err = {}, Time = {}, Epochs = {}".format(\
            mse_loss, mean_norm_err, median_mean_norm_err, std_mean_norm_err, mean_t_conv, mean_ep_conv))

    if not os.path.isfile('./out_hyp.csv'):
        out = open('out_hyp.csv', 'w')
        out.write('Nodes|Communities|N samples|Batch size|' +
                    'F in|F out|K in|K out|C|Learning Rate|Non Lin|Median|' +
                    'MSE loss|Mean norm err|Median mean norm err|STD mean norm err|' +
                    'Mean t convergence|Mean epochs convergence\n')
    else:
        out = open('out_hyp.csv', 'a')
    out.write("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n".format(
                    N, k, N_samples, batch_size,
                    Fi, Fo, Ki, Ko, C, learning_rate, nonlin_s, median,
                    mse_loss, mean_norm_err, median_mean_norm_err, std_mean_norm_err,
                    mean_t_conv, mean_ep_conv))
    out.close()

    # print("--------------------------------------Ended simulation--------------------------------------")
    # print("2G difussed deltas architecture parameters")
    # print("Graph: N = {}; c = {}".format(str(N), str(c)))
    # print("Neural net: Fi = {}, Fo = {}, Ki = {}, Ko = {}".format(str(Fi), str(Fo), str(Ki), str(Ko)))
    # #print("MSE loss = {}".format(str(mse_losses)))
    # print("MSE loss mean = {}".format(np.mean(mse_losses)))
    # #print("Mean Squared Error = {}".format(str(mean_norm_errs)))
    # print("Mean Norm Error = {}".format(np.mean(mean_norm_errs)))
    # print("Median error = {}".format(np.median(mean_norm_errs)))
    # print("STD = {}".format(np.std(mean_norm_errs)))
    return np.mean(mse_losses), np.mean(mean_norm_errs), np.median(mean_norm_errs), np.std(mean_norm_errs)


if __name__ == "__main__":
    for lr in learning_rate:
    for k in K:
    for bs in batch_size:
    for f in F:
        fo = f.copy()
        fo[0] = N/2
        fo.reverse()
    for comm in c:
    for m in median:
