import os
import torch
import torch.nn as nn
import numpy as np

from cnngs_src import graphtools, datatools

from GIGO.model import Model

from GIGO import data_sets
from GIGO.arch import GIGOArch

# Consts
ARCH_INFO = True
TB_LOG = True
VERB = True

# Parameters

# Data parameters
N_samples = 2000
N_test = 200
eval_freq = 4

maxdiff = 10

num_epochs = 60
batch_size = 100
max_non_dec = None

# Graph parameters
N = 64
c = 4
p_SBM = np.array([0.8, 0.7, 0.7, 0.8])
q_SBM = np.array([[0.0, 0.2, 0.0, 0.1],
                    [0.2, 0.0, 0.08, 0.0],
                    [0.0, 0.08, 0.0, 0.05],
                    [0.1, 0.0, 0.05, 0.0]])

# NN Parameters
Ki = 3
Ko = 2
Fi = [1,4,c]
Fo = [N,32,16]
C = [16,8,1]
batch_norm = False

loss_func = nn.CrossEntropyLoss()

optimizer = "ADAM"
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
decay_rate = 0.99
nonlin = nn.ReLU
#nonlin = nn.Tanh

# Define the datamodel

dataset = data_sets.GIGOSourceLocMPSBM(N, c, p_SBM, q_SBM, N_samples, N_test, \
                                maxdiff)

archit = GIGOArch(dataset.Ngraph, dataset.Cgraph, Fi, Fo, Ki, Ko, C, nonlin, batch_norm, ARCH_INFO)

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
model_param['max_non_dec'] = max_non_dec
model_param['tb_log'] = TB_LOG
model_param['verb'] = VERB

model = Model(**model_param)
model.eval(dataset.train_data, dataset.train_labels, dataset.val_data, dataset.val_labels, dataset.test_data, dataset.test_labels)
