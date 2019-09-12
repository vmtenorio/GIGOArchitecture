import os
import torch
import torch.nn as nn
import numpy as np

from cnngs_src import graphtools, datatools

from GIGO.model import Model

from GIGO import data_sets
from GIGO.arch import GIGOArch

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

dataset_sel = "sourceloc"       # sourceloc or maxcomm

maxdiff = 10
limit_data = 25
data_sym = True

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
Fi = [1,4,c]
Fo = [N,4,1]

if dataset_sel == "sourceloc":
    loss_func = nn.CrossEntropyLoss()
elif dataset_sel == "maxcomm":
    loss_func = nn.MSELoss()

optimizer = "ADAM"
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
decay_rate = 0.9
nonlin = nn.ReLU
#nonlin = nn.Sigmoid

# Define the datamodel
if dataset_sel == "sourceloc":
    dataset = data_sets.GIGOSourceLoc(N, c, p_ii_SBM, p_ij_SBM, N_samples, \
                                maxdiff, train_test_coef)
elif dataset_sel == "maxcomm":
    dataset = data_sets.GIGOMaxComm(N, c, p_ii_SBM, p_ij_SBM, N_samples, \
                                train_test_coef, limit_data, data_sym)
else:
    raise RuntimeError

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
