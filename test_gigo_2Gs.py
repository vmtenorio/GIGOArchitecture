import os
import torch.nn as nn
import numpy as np
import time

from graph_enc_dec.model import Model, ADAM
from graph_enc_dec import data_sets
from GIGO.arch import GIGOArch

from multiprocessing import Pool, cpu_count

SEED = None
VERB = True
ARCH_INFO = True
N_CPUS = cpu_count()

# Parameters

# Data parameters
N_samples = 2000
eval_freq = 4
N_graphs = 5

L_filter = 6

num_epochs = 100
batch_size = 50
max_non_dec = 5

# Graph parameters
N = 64
k = 4
G_params = {}
G_params['type'] = data_sets.SBM
G_params['N'] = N
G_params['k'] = k
G_params['p'] = [0.6, 0.7, 0.6, 0.8]
G_params['q'] = 0.2
G_params['type_z'] = data_sets.RAND

mod_type = "link_pert"
Gs = {}
Gs['params'] = G_params
Gs['perm'] = True
if mod_type == "nodes_pert":
    Gs['pert'] = 30
else:
    Gs['pct'] = True
    if Gs['pct']:
        Gs['eps1'] = 10
        Gs['eps2'] = 10
    else:
        Gs['eps1'] = 0.1
        Gs['eps2'] = 0.3
    # Gs['pert'] = 0          # For features in

median = True

# NN Parameters
Ki = 2
Ko = 2
Fi = [1, int(N/2), N]
Fo = [N, int(N/2), int(N/4)]
C = [Fo[-1], int(N/4), 1]
nonlin_s = "tanh"
if nonlin_s == "relu":
    nonlin = nn.ReLU
elif nonlin_s == "tanh":
    nonlin = nn.Tanh
elif nonlin_s == "sigmoid":
    nonlin = nn.Sigmoid
batch_norm = False

loss_func = nn.MSELoss()

optimizer = ADAM
learning_rate = 0.01
decay_rate = 0.99

model_param = {}

model_param['opt'] = optimizer
model_param['learning_rate'] = learning_rate
model_param['decay_rate'] = decay_rate
model_param['loss_func'] = loss_func
model_param['epochs'] = num_epochs
model_param['batch_size'] = batch_size
model_param['eval_freq'] = eval_freq
model_param['max_non_dec'] = max_non_dec
model_param['verbose'] = VERB


def test_model(Gs, N_samples, L_filter, Fi, Fo, Ki, Ko, C, nonlin, model_param):
    Gx, Gy = data_sets.perturbated_graphs(Gs['params'], Gs['eps1'], Gs['eps2'],
                                          perm=Gs['perm'], seed=SEED)

    # Define the data model
    data = data_sets.LinearDS2GSLinksPert(Gx, Gy, N_samples, L_filter, G_params['k']) # Last argument is n_delts
    data.to_unit_norm()
    # data.add_noise(signals['noise'], test_only=signals['test_only']) # For future tests
    data.to_tensor()

    Gx.compute_laplacian('normalized')
    Gy.compute_laplacian('normalized')

    # With L
    archit = GIGOArch(Gx.L.todense(), Gy.L.todense(), Fi, Fo, Ki, Ko, C, nonlin, batch_norm, ARCH_INFO)
    # With A
    # archit = GIGOArch(Gx.W.todense().astype(int), Gy.W.todense().astype(int), Fi, Fo, Ki, Ko, C, nonlin, batch_norm, ARCH_INFO)

    model_param['arch'] = archit

    model = Model(**model_param)
    t_init = time.time()
    epochs, _, _ = model.fit(data.train_X, data.train_Y, data.val_X, data.val_Y)
    t_conv = time.time() - t_init
    mean_err, med_err, mse = model.test(data.test_X, data.test_Y)

    return mse, med_err, mean_err, model.count_params(), t_conv, epochs


if __name__ == '__main__':
    pool = Pool(processes=N_CPUS)
    results = []
    for ng in range(N_graphs):
        print("Started test " + str(ng))
        results.append(pool.apply_async(test_model,\
                       args=[Gs, N_samples, L_filter, Fi, Fo, Ki, Ko, C, nonlin, model_param]))

    mse_losses = np.zeros(N_graphs)
    mean_errs = np.zeros(N_graphs)
    med_errs = np.zeros(N_graphs)
    t_conv = np.zeros(N_graphs)
    epochs_conv = np.zeros(N_graphs)
    for ng in range(N_graphs):
        # No problem in overriding n_params, as it has always the same value
        mse_losses[ng], med_errs[ng], mean_errs[ng], n_params, t_conv[ng], epochs_conv[ng] = results[ng].get()

    mse_loss = np.median(mse_losses)
    mean_norm_err = np.mean(med_errs)
    median_mean_norm_err = np.median(med_errs)
    std_mean_norm_err = np.std(med_errs)
    mean_t_conv = round(np.mean(t_conv), 6)
    mean_ep_conv = np.mean(epochs_conv)
    print("--------------------------------------Ended simulation--------------------------------------")
    print("2G difussed deltas architecture parameters")
    print("Graph: N = {}; k = {}".format(str(N), str(k)))
    print("Fin: {}, Fout: {}, Kin: {}, Kout: {}, C: {}".format(Fi, Fo, Ki, Ko, C))
    print("Non lin: " + str(nonlin))
    print("N params: " + str(n_params))
    # print("MSE loss = {}".format(str(mse_losses)))
    print("MSE loss mean = {}".format(mse_loss))
    # print("Mean Squared Error = {}".format(str(mean_norm_errs)))
    print("Mean Norm Error = {}".format(mean_norm_err))
    print("Median error = {}".format(median_mean_norm_err))
    print("STD = {}".format(std_mean_norm_err))
    print("Until convergence: Time = {} - Epochs = {}".format(mean_t_conv, mean_ep_conv))
    if not os.path.isfile('./out.csv'):
        f = open('out.csv', 'w')
        f.write('Nodes|Communities|N samples|Batch size|' +
                'F in|F out|K in|K out|C|Learning Rate|Non Lin|Median|' +
                'MSE loss|Mean norm err|Median mean norm err|STD mean norm err|' +
                'Mean t convergence|Mean epochs convergence\n')
    else:
        f = open('out.csv', 'a')
    f.write("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n".format(
                    N, k, N_samples, batch_size,
                    Fi, Fo, Ki, Ko, C, learning_rate, nonlin_s, median,
                    mse_loss, mean_norm_err, median_mean_norm_err, std_mean_norm_err,
                    mean_t_conv, mean_ep_conv))
    f.close()
