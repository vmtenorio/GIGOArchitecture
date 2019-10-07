import os
import torch
import torch.nn as nn
import numpy as np

from cnngs_src import graphtools, datatools

from GIGO.model import Model

from graph_enc_dec import data_sets
from GIGO.arch import GIGOArch

SEED = 15
TB_LOG = True

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

# NN Parameters
Ki = 2
Ko = 2
Fi = [1,16,N]
Fo = [N,16,1]

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
Fi = [[1, 16, N],
        [1,N],
        [1, N/4, N/2, N]]       # Would need to check N is a multiple of 2 and 4

Fo = [[N, 16, 1],
        [N, 1],
        [N, N/2, N/4, 1]]       # Would need to check N is a multiple of 2 and 4

learning_rate = [0.1, 0.01, 0.001, 0.05]

Ki = [2,3,4]
Ko = [2,3,4]

batch_size = [10,50,100,200]

N = [64,128,256]

c = [4,8,16]

median = [True, False]

def test_arch(G_params, model_param, median, Fi, Fo, Ki, Ko):

    for ng in range(N_graphs):

        Gx, Gy = data_sets.perturbated_graphs(G_params, eps1, eps2, seed=SEED)

        # Define the data model
        data = data_sets.LinearDS2GS(Gx, Gy, N_samples, L_filter, G_params['k'], median=median)
        data.to_unit_norm()

        Gx.compute_laplacian('normalized')
        Gy.compute_laplacian('normalized')

        archit = GIGOArch(Gx.L.todense(), Gy.L.todense(), Fi, Fo, Ki, Ko, nonlin)

        model_param['arch'] = archit

        model = Model(**model_param)
        mse_loss, _, mean_norm_err = model.eval(data.train_X, data.train_Y, data.val_X, data.val_Y, data.test_X, data.test_Y)
        mse_losses.append(mse_loss)
        mean_norm_errs.append(mean_norm_err)

        print("--------------------------------------Ended simulation--------------------------------------")
        print("2G difussed deltas architecture parameters")
        print("Graph: N = {}; c = {}".format(str(N), str(c)))
        print("Neural net: Fi = {}, Fo = {}, Ki = {}, Ko = {}".format(str(Fi), str(Fo), str(Ki), str(Ko)))
        #print("MSE loss = {}".format(str(mse_losses)))
        print("MSE loss mean = {}".format(np.mean(mse_losses)))
        #print("Mean Squared Error = {}".format(str(mean_norm_errs)))
        print("Mean Norm Error = {}".format(np.mean(mean_norm_errs)))
        print("Median error = {}".format(np.median(mean_norm_errs)))
        print("STD = {}".format(np.std(mean_norm_errs)))
        return np.mean(mse_losses), np.mean(mean_norm_errs), np.median(mean_norm_errs), np.str(mean_norm_errs)

if __name__ == "__main__":
    # Think this better
    outfile = open('out.csv', 'w')
    for fi in Fi:
        for fo in Fo:
            for lr in learning_rate:
                for ki in Ki:
                    for ko in Ko:
                        for bs in batch_size:
                            for n in N:
                                for k in c:
                                    for m in median:
                                        G_params['N'] = n
                                        G_params['k'] = k

                                        model_param['learning_rate'] = lr
                                        model_param['batch_size'] = bs

                                        # More parameters to be added
                                        mse_loss, mean_norm_err, median_mean_norm_err, std_mean_norm_err = test_arch(G_params, model_param, m, fi, fo, ki, ko)

                                        outfile.write("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}".format(\
                                                        fi,fo,lr,ki,ko,bs,n,k,m, \
                                                        mse_loss,mean_norm_err, median_mean_norm_err, std_mean_norm_err))
    outfile.close()