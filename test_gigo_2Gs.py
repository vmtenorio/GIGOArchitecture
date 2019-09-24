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
N_samples = 10000
eval_freq = 4

L_filter = 5

num_epochs = 40
batch_size = 100

# Graph parameters
N = 10
c = 2
G_params = {}
G_params['type'] = data_sets.SBM #SBM or ER
G_params['N'] = N
G_params['k'] = c
G_params['p'] = 0.8
G_params['q'] = 0.2
G_params['type_z'] = data_sets.CONT
eps1 = 0.1
eps2 = 0.3

Gx, Gy = data_sets.perturbated_graphs(G_params, eps1, eps2, seed=SEED)

# NN Parameters
arch_type = 'basic'
Ki = 3
Ko = 2
L = 2
Fi = [1,4,N]
Fo = [N,4,1]

loss_func = nn.MSELoss()

optimizer = "ADAM"
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
decay_rate = 0.9
nonlin = nn.ReLU
#nonlin = nn.Sigmoid

# Define the data model
data = data_sets.DiffusedSparse2GS(Gx, Gy, N_samples, L_filter, G_params['k'])
data.to_unit_norm()

archit = GIGOArch(Gx.W.todense().astype(int), Gy.W.todense().astype(int), Fi, Fo, Ki, Ko, nonlin)

model_param = {}

model_param['arch'] = archit
model_param['optimizer'] = optimizer
model_param['learning_rate'] = learning_rate
model_param['beta1'] = beta1
model_param['beta2'] = beta2
model_param['decay_rate'] = decay_rate
model_param['loss_func'] = loss_func
model_param['num_epochs'] = num_epochs
model_param['batch_size'] = batch_size
model_param['eval_freq'] = eval_freq
model_param['tb_log'] = TB_LOG

model = Model(**model_param)
# Transpose them as they are N x T, and the architecture works with T x N
print(data.test_X.shape)
model.eval(data.train_X.T, data.train_Y.T, data.test_X, data.test_Y)
