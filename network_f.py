import os
import torch
import torch.nn as nn
import numpy as np

from cnngs_src import graphtools, datatools

from model import Model
from data_sets import FlightData
import arch

# Consts
if os.name == 'nt':
    MAIN_PATH = '\\\\192.168.1.35\\Share\\Aero_TFG\\'
    slash = '\\'
else:
    MAIN_PATH = '/shared/Aero_TFG/'
    slash = '/'
FILES_PATH = MAIN_PATH + 'DataProc' + slash
LAST_EXEC = FILES_PATH + '20190609' + slash
DEBUG = True
TB_LOG = True

# Parameters

num_epochs = 40
batch_size = 100
eval_freq = 4

# NN Parameters
K = 5
L = 2
F = [2,32,32,8,1]

M = [32]

loss_func = nn.MSELoss()
#loss_func = nn.CrossEntropyLoss()
optimizer = "ADAM"
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
decay_rate = 0.9
nonlin = nn.LeakyReLU

# Generate data object
data = FlightData(LAST_EXEC + 'adj_mat.csv', LAST_EXEC + 'dep_delay.csv', 0.8, 2)

M.append(data.N_nodes)

archit = arch.BasicArch(data.S, F, K, M, nonlin)

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
model.eval(data.train_data, data.train_labels, data.test_data, data.test_labels)
