import os
import torch
import torch.nn as nn
import numpy as np

from cnngs_src import graphtools, datatools

from model import Model

from data_sets import GIGOMaxComm
from arch import GIGOArch

# Consts
if os.name == 'nt':
    MAIN_PATH = '\\\\192.168.1.35\\Share\\Aero_TFG\\'
    slash = '\\'
else:
    MAIN_PATH = '/shared/Aero_TFG/'
    slash = '/'
FILES_PATH = MAIN_PATH + 'DataProc' + slash
DEBUG = True
TB_LOG = True

# Parameters

# Data parameters
N_samples = 10000
eval_freq = 4
train_test_coef = 0.8
limit_data = 25

noise_train = False
noise_test = False
sigma_data = np.sqrt(1e-4)

num_epochs = 40
batch_size = 100

# Graph parameters
N = 128 
select_graph = 'SBM'
p_ER = 0.4
c = 8
N_classes = c
p_ii_SBM = 0.8
p_ij_SBM = 0.2

# NN Parameters
arch_type = 'basic'
Ki = 5
Ko = 2
L = 2
Fi = [1,32,c]
Fo = [N,2,1]

loss_func = nn.MSELoss()
optimizer = "ADAM"
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
decay_rate = 0.9
nonlin = nn.LeakyReLU

# Define the datamodel
dataset = GIGOMaxComm(N, c, p_ii_SBM, p_ij_SBM, N_samples, \
                                train_test_coef, limit_data)

archit = GIGOArch(dataset.Ngraph, dataset.Cgraph, Fi, Fo, Ki, Ko, nonlin)

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
model.eval(dataset.train_data, dataset.train_labels, dataset.test_data, dataset.test_labels)
