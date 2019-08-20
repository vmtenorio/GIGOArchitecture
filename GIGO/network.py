import os
import torch
import torch.nn as nn
import numpy as np

from cnngs_src import graphtools, datatools

from model import Model

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
maxdiff = 25
N_test = 400

noise_train = False
noise_test = False
sigma_data = np.sqrt(1e-4)

num_epochs = 40
batch_size = 100

# Graph parameters
N = 150
select_graph = 'SBM'
p_ER = 0.4
c = 5
N_classes = c
p_ii_SBM = 0.8
p_ij_SBM = 0.2

# NN Parameters
arch_type = 'basic'
K = 5
L = 2
F = [1,32,32]

M = [32, N_classes]

loss_func = nn.CrossEntropyLoss()
optimizer = "ADAM"
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
decay_rate = 0.9
nonlin = nn.ReLU

# Graph creation
if select_graph == 'SBM':
    p_SBM = p_ij_SBM * np.ones(c)
    p_SBM = np.abs(p_ii_SBM-p_ij_SBM)*np.eye(c)+p_SBM
    N_SBM = int(np.floor(N/c)) * np.ones([c,1])
    missing = (N - np.sum(N_SBM)).astype(np.int)
    for it in range(missing):
        N_SBM[it] = N_SBM[it]+1
    N_SBM = N_SBM.astype(np.int)
    A = graphtools.create_SBM(N_SBM,p_SBM)
elif select_graph == 'ER':
    A = graphtools.create_ER(N,p_ER)

S = graphtools.norm_graph(A)

arch_param = {}

arch_param['S'] = S
arch_param['arch_type'] = arch_type
arch_param['F'] =  F
arch_param['K'] = K
arch_param['M'] = M
arch_param['nonlin'] = nonlin
arch_param['optimizer'] = optimizer
arch_param['learning_rate'] = learning_rate
arch_param['beta1'] = beta1
arch_param['beta2'] = beta2
arch_param['decay_rate'] = decay_rate
arch_param['loss_func'] = loss_func
arch_param['num_epochs'] = num_epochs
arch_param['batch_size'] = batch_size
arch_param['eval_freq'] = eval_freq
arch_param['tb_log'] = TB_LOG

#Data Definition
train_labels = np.ceil(N_classes*np.random.rand(N_samples))
train_data = datatools.create_samples(S, train_labels, maxdiff)
if noise_train:
    w_train = sigma_data*np.random.randn(N,N_samples)
    train_data = train_data + w_train

test_labels = np.ceil(N_classes*np.random.rand(N_test))
test_data = datatools.create_samples(S, test_labels, maxdiff)
if noise_test:
    w_test = sigma_data*np.random.randn(N,N_test)
    test_data += w_test

train_labels = train_labels.astype(np.int64)-1
test_labels = test_labels.astype(np.int64)-1
train_data = train_data.transpose()
test_data = test_data.transpose()

# Turn data into tensors
train_data = torch.FloatTensor(train_data)
train_labels = torch.LongTensor(train_labels) # CrossEntropyLoss requires Long Type
test_data = torch.FloatTensor(test_data)
test_labels = torch.LongTensor(test_labels)

model = Model(**arch_param)
model.eval(train_data, train_labels, test_data, test_labels)
