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
N_graphs = 1

L_filter = 5

num_epochs = 40
batch_size = 100
max_non_dec = 5

# Graph parameters
N = 128
c = 4
G_params = {}
G_params['type'] = data_sets.SBM #SBM or ER
G_params['N'] = N
G_params['k'] = c
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
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
decay_rate = 0.99
nonlin = nn.Tanh

model_param = {}

model_param['optimizer'] = optimizer
model_param['learning_rate'] = learning_rate
model_param['beta1'] = beta1
model_param['beta2'] = beta2
model_param['decay_rate'] = decay_rate
model_param['loss_func'] = loss_func
model_param['num_epochs'] = num_epochs
model_param['batch_size'] = batch_size
model_param['eval_freq'] = eval_freq
model_param['max_non_dec'] = max_non_dec
model_param['tb_log'] = TB_LOG

mse_losses = []
mean_norm_errs = []

for ng in range(N_graphs):

    Gx, Gy = data_sets.perturbated_graphs(G_params, eps1, eps2, seed=SEED)

    # Define the data model
    data = data_sets.LinearDS2GS(Gx, Gy, N_samples, L_filter, G_params['k'])
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
#print("MSE loss = {}".format(str(mse_losses)))
print("MSE loss mean = {}".format(np.mean(mse_losses)))
#print("Mean Squared Error = {}".format(str(mean_norm_errs)))
print("Mean Norm Error = {}".format(np.mean(mean_norm_errs)))
print("Median error = {}".format(np.median(mean_norm_errs)))
print("STD = {}".format(np.std(mean_norm_errs)))
