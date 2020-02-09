import os
import time
import torch.nn as nn
import numpy as np
import sys
sys.path.append('..')

from graph_enc_dec.model import Model, ADAM

from graph_enc_dec import data_sets
from GIGO.arch import BasicArch, MLP, ConvNN

from multiprocessing import Pool, cpu_count

TB_LOG = False
VERB = False
ARCH_INFO = False
N_CPUS = cpu_count()

# Parameters

signals = {}
signals['N_samples'] = 2000
signals['N_graphs'] = 20
signals['L_filter'] = 6
signals['noise'] = 0
signals['test_only'] = True

signals['perm'] = True
signals['median'] = True
signals['pert'] = 30

# Graph parameters
G_params = {}
G_params['type'] = data_sets.SBM
G_params['N'] = N = 256
G_params['k'] = k = 4
G_params['p'] = 0.3
G_params['q'] = [[0, 0.0075, 0, 0.0],
                 [0.0075, 0, 0.004, 0.0025],
                 [0, 0.004, 0, 0.005],
                 [0, 0.0025, 0.005, 0]]
G_params['type_z'] = data_sets.RAND
signals['g_params'] = G_params

CONV1 = [{'f_enc': [1, 2, 2, 2, 3],  # 10
          'kernel_enc': 11,
          'f_dec': [3, 2, 2, 1],
          'kernel_dec': 11},
         {'f_enc': [1, 2, 2, 2, 3],  # 20
          'kernel_enc': 11,
          'f_dec': [3, 3, 1],
          'kernel_dec': 11},
         {'f_enc': [1, 2, 2, 2, 2, 3],  # 30
          'kernel_enc': 11,
          'f_dec': [3, 2, 1],
          'kernel_dec': 11},
         {'f_enc': [1, 2, 2, 2, 3, 3],  # 40
          'kernel_enc': 11,
          'f_dec': [3, 1],
          'kernel_dec': 11},
         {'f_enc': [1, 1, 1, 1, 2, 2, 2, 3],  # 50
          'kernel_enc': 11,
          'f_dec': [3, 2, 1],
          'kernel_dec': 11}]

CONV2 = [{'f_enc': [1, 1, 1, 2, 2],  # 10
          'kernel_enc': 11,
          'f_dec': [2, 1, 1, 1],
          'kernel_dec': 11},
         {'f_enc': [1, 1, 1, 2, 2],  # 20
          'kernel_enc': 11,
          'f_dec': [2, 1, 1],
          'kernel_dec': 11},
         {'f_enc': [1, 1, 2, 2, 2],  # 30
          'kernel_enc': 11,
          'f_dec': [2, 1],
          'kernel_dec': 11},
         {'f_enc': [1, 1, 1, 1, 2, 2],  # 40
          'kernel_enc': 11,
          'f_dec': [2, 1],
          'kernel_dec': 11},
         {'f_enc': [1, 1, 1, 1, 1, 2, 2],  # 50
          'kernel_enc': 11,
          'f_dec': [2, 1],
          'kernel_dec': 11}]

EXPS = [
    {
        'arch_type': 'linear',
        'F': [N, None],
        'K': "N/A",
        'M': "N/A",
        'nonlin': None
    },
    {
        'arch_type': 'mlp',
        'F': [N, None],
        'K': "N/A",
        'M': "N/A",
        'nonlin': nn.Tanh
    },
    {
        'arch_type': 'mlp',
        'F': [N, 2, None],
        'K': "N/A",
        'M': "N/A",
        'nonlin': nn.Tanh
    },
    {
        'arch_type': 'conv',
        'F': [1, 4, 8, 8, 4, 1],
        'K': 3,
        'M': [1, None],
        'nonlin': nn.Tanh
    },
    {
        'arch_type': 'basic',
        'F': [1, 4, 8, 8, 4, 1],
        'K': 3,
        'M': [1, None],
        'nonlin': nn.Tanh
    }
]

# Model parameters
model_params = {}
model_params['opt'] = ADAM
model_params['learning_rate'] = 0.05
model_params['decay_rate'] = 0.99
model_params['loss_func'] = nn.MSELoss()
model_params['epochs'] = 200
model_params['batch_size'] = 50
model_params['eval_freq'] = 4
model_params['max_non_dec'] = 10
model_params['verbose'] = VERB

# Test n1 paper -- Deleting nodes
param_list = [10, 20, 30, 40, 50]

# Test n2 paper -- Adding noise
# param_list = [0, .025, .05, 0.75, .1]

def test_model(id, signals, nn_params, model_params):
    Gx, Gy = data_sets.nodes_perturbated_graphs(signals['g_params'], signals['pert'],
                                                perm=signals['perm'])

    # Define the data model
    data = data_sets.LinearDS2GSNodesPert(Gx, Gy, signals['N_samples'],
                                          signals['L_filter'], signals['g_params']['k'],  # k is n_delts
                                          median=signals['median'])
    data.to_unit_norm()
    data.add_noise(signals['noise'], test_only=signals['test_only'])
    data.to_tensor()

    if nn_params['arch_type'] == "basic":
        Gx.compute_laplacian('normalized')
        archit = BasicArch(Gx.L.todense(), nn_params['F'], nn_params['K'], nn_params['M'], nn_params['nonlin'], ARCH_INFO)
    elif nn_params['arch_type'] == "mlp":
        archit = MLP(nn_params['F'], nn_params['nonlin'], ARCH_INFO)
    elif nn_params['arch_type'] == "conv":
        archit = ConvNN(N, nn_params['F'], nn_params['K'], nn_params['nonlin'], nn_params['M'], ARCH_INFO)
    elif nn_params['arch_type'] == "linear":
        archit = MLP(nn_params['F'], nn_params['nonlin'], ARCH_INFO)
    else:
        raise RuntimeError("arch_type has to be either basic, mlp or conv")

    model_params['arch'] = archit

    model = Model(**model_params)
    print("Numero de parametros " + str(model.count_params()))
    t_init = time.time()
    epochs, _, _ = model.fit(data.train_X, data.train_Y, data.val_X, data.val_Y)
    t_conv = time.time() - t_init
    mean_err, med_err, mse = model.test(data.test_X, data.test_Y)

    print("DONE {}: MSE={} - Mean Err={} - Median Err={} - Params={} - t_conv={} - epochs={}".format(
        id, mse, mean_err, med_err, model.count_params(), round(t_conv, 4), epochs
    ))
    return mse, mean_err, med_err, model.count_params(), t_conv, epochs

def test_exp(signals, nn_params, model_params):
    print("Testing: Arch type = {}, F = {}, K = {}, M = {}, Noise = {}, Pert = {}".format( \
        nn_params['arch_type'], nn_params['F'], nn_params['K'], nn_params['M'],
        signals['noise'], signals['pert']))

    pool = Pool(processes=N_CPUS)

    results = []
    for ng in range(signals['N_graphs']):
        results.append(pool.apply_async(test_model,
                                        args=[ng, signals, nn_params, model_params]))

    mean_errs = np.zeros(signals['N_graphs'])
    mse_losses = np.zeros(signals['N_graphs'])
    med_errs = np.zeros(signals['N_graphs'])
    t_conv = np.zeros(signals['N_graphs'])
    epochs_conv = np.zeros(signals['N_graphs'])
    for ng in range(signals['N_graphs']):
        # No problem in overriding n_params, as it has always the same value
        mse_losses[ng], mean_errs[ng], med_errs[ng], n_params, t_conv[ng], epochs_conv[ng] = results[ng].get()

    mse_loss = np.median(mse_losses)
    mean_err = np.median(mean_errs)
    median_err = np.median(med_errs)
    std_err = np.std(med_errs)
    mean_t_conv = round(np.mean(t_conv), 6)
    mean_ep_conv = np.mean(epochs_conv)
    print("-----------------------------------------------------------------------------------")
    print("DONE Test: MSE={} - Mean Err={} - Median Err={} - Params={} - t_conv={} - epochs={}".format(
        mse_loss, mean_err, median_err, n_params, mean_t_conv, mean_ep_conv
    ))
    print("-----------------------------------------------------------------------------------")

    if not os.path.isfile('./out_basic.csv'):
        outf = open('out_basic.csv', 'w')
        outf.write('Experiment|Nodes|Communities|N samples|N graphs|' +
                   'Perturbation|L filter|Noise|' +
                   'F|K|M|' +
                   'Non Lin|' +
                   'Batch size|Learning Rate|' +
                   'Num Params|MSE loss|Mean err|Median err|STD Median err|' +
                   'Mean t convergence|Mean epochs convergence\n')
    else:
        outf = open('out_basic.csv', 'a')
    outf.write("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n".format(
        "NodesPert", N, k, signals['N_samples'], signals['N_graphs'],
        signals['pert'], signals['L_filter'], signals['noise'],
        nn_params['F'], nn_params['K'], nn_params['M'],
        nn_params['nonlin'],
        model_params['batch_size'], model_params['learning_rate'],
        n_params, mse_loss, mean_err, median_err, std_err,
        mean_t_conv, mean_ep_conv))
    outf.close()

    return np.median(med_errs)  # Keeping the one with the best median error


if __name__ == '__main__':
    for p in param_list:
        signals['pert'] = p
        Nout = N - p
        # signals['noise'] = p
        for exp in EXPS:

            if exp['arch_type'] == "basic":
                exp['M'][-1] = Nout
            elif exp['arch_type'] == "mlp":
                exp['F'][-1] = Nout
            elif exp['arch_type'] == "conv":
                exp['M'][-1] = Nout
            elif exp['arch_type'] == "linear":
                exp['F'][-1] = Nout
            else:
                raise RuntimeError("arch_type has to be either basic, mlp or conv")

            test_exp(signals, exp, model_params)
