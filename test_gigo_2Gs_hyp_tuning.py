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
N_CPUS = cpu_count()

# Parameters

# Data parameters
N_samples = 2000
eval_freq = 4
N_graphs = 10

L_filter = 5

num_epochs = 40
batch_size = 100
max_non_dec = 5

# Graph parameters
G_params = {}
G_params['type'] = data_sets.SBM #SBM or ER
G_params['p'] = 0.8
G_params['q'] = 0.2
G_params['type_z'] = data_sets.CONT
eps1 = 0.1
eps2 = 0.3

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

mse_losses = []
mean_norm_errs = []

# Hyperparameters tuning

learning_rate = [0.05]

K = [2,3,4]

batch_size = [100]

N = [64]

# For number of features see below, as it depends on the number of nodes

c = [4]

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
    return mse_loss, mean_norm_err

def test_arch(G_params, model_param, median, Fi, Fo, Ki, Ko):
    pool = Pool(processes=N_CPUS)

    results = []
    for ng in range(N_graphs):
        results.append(pool.apply_async(run_arch,\
                        args=[G_params, model_param, median, Fi, Fo, Ki, Ko]))

    mean_norm_errs = np.zeros(N_graphs)
    mse_losses = np.zeros(N_graphs)
    for ng in range(N_graphs):
        mse_losses[ng], mean_norm_errs[ng]  = results[ng].get()

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
    # Think this better
    outfile = open('out.csv', 'a')
    #outfile.write("Fin|Fout|Learning Rate|Kin|Kout|Batch Size|N nodes|N comm|Median|MSE loss|Mean Norm Err|Median Mean Norm Err|STD Mean Norm Err|Time Taken\n")
    for lr in learning_rate:
        for k in K:
            for bs in batch_size:
                for n in N:
                    F = [[1, int(n/2), n],
                            [1, n],
                            [1, int(n/4), int(n/2), n]]       # Would need to check N is a multiple of 2 and 4
                    for f in F:
                        fo = f.copy()
                        fo.reverse()
                        for comm in c:
                            for m in median:
                                G_params['N'] = n
                                G_params['k'] = comm

                                model_param['learning_rate'] = lr
                                model_param['batch_size'] = bs
                                print("Testing: N = {}, c = {}, F = {}, K = {}, BS = {}, LR = {}, Median = {}".format(\
                                        n, comm, f, k, bs, lr, m))

                                # More parameters to be added
                                t_init = time()
                                mse_loss, mean_norm_err, median_mean_norm_err, std_mean_norm_err = test_arch(G_params, model_param, m, f, fo, k, k)
                                t_spent = time() - t_init

                                print("Results: MSE loss = {}, Mean Norm Err = {}, Median Norm Err = {}, STD Norm Err = {}, Time = {}".format(\
                                        mse_loss, mean_norm_err, median_mean_norm_err, std_mean_norm_err, t_spent))

                                outfile.write("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n".format(\
                                                f,fo,lr,k,k,bs,n,k,m, \
                                                mse_loss,mean_norm_err, median_mean_norm_err, std_mean_norm_err, \
                                                round(t_spent, 2)))
    outfile.close()
