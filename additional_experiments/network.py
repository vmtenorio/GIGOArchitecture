import os
import torch
import torch.nn as nn
import numpy as np

from cnngs_src import graphtools, datatools

from GIGO.model import Model
from GIGO import data_sets
from GIGO import arch

# Consts
DEBUG = True
TB_LOG = True

# Parameters

# Data parameters
N_samples = 10000
eval_freq = 4
maxdiff = 25
train_test_coef = 0.8

noise_train = False
noise_test = False
sigma_data = np.sqrt(1e-4)

num_epochs = 40
batch_size = 100
max_non_dec = 5

# Graph parameters
N = 64
select_graph = 'SBM'
p_ER = 0.4
c = 4
p_ii_SBM = 0.8
p_ij_SBM = 0.2

# NN Parameters
K = 3
L = 2
F = [1,32,32]

M = [32, N] # I'm getting the comm that originated the delta

loss_func = nn.CrossEntropyLoss()
optimizer = "ADAM"
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
decay_rate = 0.9
nonlin = nn.Tanh

#Dataset Definition
data = data_sets.SourceLocalization(N, c, p_ii_SBM, p_ij_SBM, N_samples, \
                            maxdiff, train_test_coef)

# Arch definition
arch = arch.BasicArch(data.S, F, K, M, nonlin)

model_param = {}

model_param['arch'] = arch
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

model = Model(**model_param)
model.eval(data.train_data, data.train_labels, data.test_data, data.test_labels, data.test_data, data.test_labels)
