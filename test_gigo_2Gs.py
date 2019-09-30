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

L_filter = 5

num_epochs = 40
batch_size = 100

# Graph parameters
N = 64
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

Gx, Gy = data_sets.perturbated_graphs(G_params, eps1, eps2, seed=SEED)

# NN Parameters
arch_type = 'basic'
Ki = 3
Ko = 2
L = 2
Fi = [1,N]
Fo = [N,1]

loss_func = nn.MSELoss()

optimizer = "ADAM"
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
decay_rate = 0.99
nonlin = nn.Tanh

# Define the data model
data = data_sets.DiffusedSparse2GS(Gx, Gy, N_samples, L_filter, G_params['k'])
data.to_unit_norm()

Lx = graphtools.norm_graph(Gx.W.todense())
Ly = graphtools.modify_graph(Gx.W.todense(), N)

Ly = graphtools.norm_graph(Ly)

archit = GIGOArch(Lx, Ly, Fi, Fo, Ki, Ko, nonlin)

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
model.eval(data.train_X, data.train_Y, data.test_X, data.test_Y)
