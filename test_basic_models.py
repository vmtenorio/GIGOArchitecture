import os
import torch
import torch.nn as nn
import numpy as np

from cnngs_src import graphtools, datatools

from GIGO.model import Model

from graph_enc_dec import data_sets
from GIGO.arch import BasicArch, MLP, ConvNN

from multiprocessing import Pool, cpu_count

SEED = None
TB_LOG = False
VERB = True
ARCH_INFO = True
N_CPUS = cpu_count()

# Parameters
arch_type = "basic" # basic, mlp, conv, linear

# Data parameters
N_samples = 10000
eval_freq = 4
N_graphs = 12

L_filter = 5

num_epochs = 200
batch_size = 100
max_non_dec = 5

# Graph parameters
N = 32
k = 4
G_params = {}
G_params['type'] = data_sets.SBM #SBM or ER
G_params['N'] = N
G_params['k'] = k
G_params['p'] = 0.8
G_params['q'] = 0.2
G_params['type_z'] = data_sets.CONT
pct = True
if pct:
    eps1 = 5
    eps2 = 5
else:
    eps1 = 0.1
    eps2 = 0.3
median = True

# NN Parameters
if arch_type == "basic":
    F = [1,int(N/2),N]
    M = [N*F[-1],2*N,N]
    K = 2
elif arch_type == "conv":
    F = [1, int(N/2), N]
    K = 2
    M = "N/A"
elif arch_type == "mlp":
    F = [N, N, N]
    K = "N/A"
    M = "N/A"
elif arch_type == "linear":
    F = [N, N, N]
    K = "N/A"
    M = "N/A"

loss_func = nn.MSELoss()

optimizer = "ADAM"
learning_rate = 0.03
beta1 = 0.9
beta2 = 0.999
decay_rate = 0.99
nonlin_s = "relu"
if nonlin_s == "relu":
    nonlin = nn.ReLU
elif nonlin_s == "tanh":
    nonlin = nn.Tanh
elif nonlin_s == "sigmoid":
    nonlin = nn.Sigmoid

if arch_type == "linear":
    nonlin_s = "None"
    nonline = None

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
model_param['verb'] = VERB

def test_model(G_params, eps1, eps2, pct, N_samples, L_filter, arch_type, F, K, nonlin, model_param):
    Gx, Gy = data_sets.perturbated_graphs(G_params, eps1, eps2, pct=pct, seed=SEED)

    # Define the data model
    data = data_sets.LinearDS2GS(Gx, Gy, N_samples, L_filter, G_params['k'])
    data.to_unit_norm()

    Gx.compute_laplacian('normalized')
    #Gy.compute_laplacian('normalized')

    if arch_type == "basic":
        archit = BasicArch(Gx.L.todense(), F, K, M, nonlin, ARCH_INFO)
    elif arch_type == "mlp":
        archit = MLP(F, nonlin, ARCH_INFO)
    elif arch_type == "conv":
        archit = ConvNN(N, F, K, nonlin, ARCH_INFO)
    elif arch_type == "linear":
        archit = MLP(F, nonlin, ARCH_INFO)
    else:
        raise RuntimeError("arch_type has to be either basic, mlp or conv")

    model_param['arch'] = archit

    model = Model(**model_param)
    mse_loss, _, mean_norm_err = model.eval(data.train_X, data.train_Y, data.val_X, data.val_Y, data.test_X, data.test_Y)

    return mse_loss, mean_norm_err, archit.n_params, model.t_conv, model.epochs_conv

if __name__ == '__main__':
    pool = Pool(processes=N_CPUS)
    results = []
    for ng in range(N_graphs):
        print("Started test " + str(ng))
        results.append(pool.apply_async(test_model,\
                        args=[G_params, eps1, eps2, pct, N_samples, L_filter, arch_type, F, K, nonlin, model_param]))

    mean_norm_errs = np.zeros(N_graphs)
    mse_losses = np.zeros(N_graphs)
    t_conv = np.zeros(N_graphs)
    epochs_conv = np.zeros(N_graphs)
    for ng in range(N_graphs):
        # No problem in overriding n_params, as it has always the same value
        mse_losses[ng], mean_norm_errs[ng], n_params, t_conv[ng], epochs_conv[ng] = results[ng].get()

    mse_loss = np.mean(mse_losses)
    mean_norm_err = np.mean(mean_norm_errs)
    median_mean_norm_err = np.median(mean_norm_errs)
    std_mean_norm_err = np.std(mean_norm_errs)
    mean_t_conv = round(np.mean(t_conv), 6)
    mean_ep_conv = np.mean(epochs_conv)
    print("--------------------------------------Ended simulation--------------------------------------")
    print("2G difussed deltas architecture parameters")
    print("Graph: N = {}; k = {}".format(str(N), str(k)))
    print("Architecture: " + arch_type)
    print("F: {}, K: {}".format(F, K))
    print("Non lin: " + str(nonlin))
    print("N params: " + str(n_params))
    #print("MSE loss = {}".format(str(mse_losses)))
    print("MSE loss mean = {}".format(mse_loss))
    #print("Mean Squared Error = {}".format(str(mean_norm_errs)))
    print("Mean Norm Error = {}".format(mean_norm_err))
    print("Median error = {}".format(median_mean_norm_err))
    print("STD = {}".format(std_mean_norm_err))
    print("Until convergence: Time = {} - Epochs = {}".format(mean_t_conv, mean_ep_conv))

    if not os.path.isfile('./out_models.csv'):
        f = open('out_models.csv', 'w')
        f.write('Architecture|Nodes|Communities|N samples|Batch size|' +
                    'F|K|M|Learning Rate|Non Lin|Median|' +
                    'MSE loss|Mean norm err|Median mean norm err|STD mean norm err|' +
                    'Mean t convergence|Mean epochs convergence\n')
    else:
        f = open('out_models.csv', 'a')
    f.write("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n".format(
                    arch_type, N, k, N_samples, batch_size,
                    F, K, M, learning_rate, nonlin_s, median,
                    mse_loss, mean_norm_err, median_mean_norm_err, std_mean_norm_err,
                    mean_t_conv, mean_ep_conv))
    f.close()
