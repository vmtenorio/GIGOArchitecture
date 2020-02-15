import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from scipy.io import savemat


PATH = './results/'


def plot_overfitting(train_err, err_val, params, show=True):
    med_train_err = np.median(train_err, axis=1)
    med_val_err = np.median(err_val, axis=1)
    _, ax = plt.subplots()
    for i in range(med_train_err.shape[1]):
        label_train = 'Train Err, P: {}'.format(params[i])
        label_val = 'Val Err, P: {}'.format(params[i])
        ax.semilogy(med_train_err[:, i], '-', label=label_train)
        ax.semilogy(med_val_err[:, i], '-', label=label_val)

    ax.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    if show:
        plt.show()


def save_results(file_pref, path, data, verbose=True):
    if not os.path.isdir(path):
        os.makedirs(path)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    if path[-1] != '/':
        path += '/'
    path = path + file_pref + timestamp
    np.save(path, data)
    if verbose:
        print('SAVED as:', os.getcwd(), path)


# NOTE: maybe create a class for print/plot?
def print_partial_results(p_n, exps, node_err, med_err):
    print('Noise:', p_n)
    for i, exp in enumerate(exps):
        print('{}. {}\n\tNode Err: {} STD: {}\n\tMedian Err: {} STD: {}'
              .format(i+1, exp, np.median(node_err[:, i]), np.std(node_err[:, i]),
                      np.median(med_err[:, i]), np.std(med_err[:, i])))


def print_results(P_n, exps, node_err, med_err):
    for i, p_n in enumerate(P_n):
        print_partial_results(p_n, exps, node_err[:, :, i], med_err[:, :, i])
        print()


def save(path, exps, node_err, med_err, Gs, signals, learning):
    if not os.path.isdir(path):
        os.makedirs(path)

    data = {}
    data['Gs'] = Gs
    data['signals'] = signals
    data['exps'] = exps
    data['node_err'] = node_err
    data['median_err'] = med_err
    data['learning'] = learning
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    file_name = path.split('/')[-1] + '_{}'.format(timestamp)
    file_path = path + '/' + file_name
    np.save(file_path, data)
    print('Saved as:', file_path)


def plot_results(x, y, fmt=[], xlabel='', legend=[]):
    # Semilogy median error
    plt.figure(figsize=(7.5, 6))
    for i in range(y.shape[0]):
        plt.semilogy(x, y[i, :], fmt[i])
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel('Median Error', fontsize=16)
    plt.gca().autoscale(enable=True, axis='x', tight=True)
    plt.gca().tick_params(labelsize=16)
    plt.grid(True)
    plt.legend(legend, prop={'size': 14})
    plt.show()


def save_mat_file(file_name, error, n_p, legend, fmt):
        file_name = './results/matfiles/' + file_name
        data = {'error': error, 'leg': legend,
                'n_p': n_p, 'fmts': fmt}
        savemat(file_name, data)
        print('Saved as:', file_name)


def plot_perturbation(file_name, params=True):
    path = PATH + 'perturbation/' + file_name + '.npy'
    data = np.load(path).item()
    med_err = np.median(data['median_err'], axis=0)
    pct = [10, 20, 30, 40]
    xlabel = '% of perturbed links'

    # if params:
    #     err = med_err[:4]
    #     fmt = ['o-', 'P-', 'X--', 'v--']
    #     legend = ['Enc/Dec Wei(192)', 'Enc/Dec NoUps(192)', 'AutoConv (210)',
    #               'AutoFC (193)']
    # else:
    #     err = med_err[4:]
    #     fmt = ['o-', 'P-', '^-', 'X--', 'v--']
    #     legend = ['Enc/Dec Wei(132)', 'Enc/Dec NoUps(132)',
    #               'Enc/Dec Wei(102)', 'AutoConv (140)', 'AutoFC (128)']

    err = med_err
    # PARA EL EXPERIMENTO QUE NO ENCUENTRO EL ARCHIVO!! 
    # fmt = ['o-', '^-', 'P-', 'X-', 'o--', '^--', 'o:', 'P--', 'X--']
    # legend = ['Enc/Dec Wei(738)', 'Enc/Dec NoUps(738)', 'AutoConv (720)',
    #           'AutoFC (769)', 'Enc/Dec Wei(554)', 'Enc/Dec NoUps(554)',
    #           'Enc/Dec Wei(132)', 'AutoConv (528)', 'AutoFC (512)']

    fmt = ['X-', 'o-', 'P-', 'o--', '^--', 'P--', 'o:', '^:', 'P:']
    legend = ['AE-FC-769', 'G-E/D-Wei-440', 'AE-CV-440',
              'G-E/D-Wei-298', 'G-E/D-NoUps-298', 'AE-CV-280',
              'G-E/D-Wei-132', 'G-E/D-NoUps-132', 'AE-CV-140']

    plot_results(pct, err, xlabel=xlabel, fmt=fmt, legend=legend)
    save_mat_file('perturbation', err, pct, legend, fmt)


def plot_noise(file_name, node_err=True):
    # path = PATH + 'diff_models/' + file_name + '.npy'
    path = PATH + 'noise/' + file_name + '.npy'
    data = np.load(path).item()
    noise = data['signals']['noise']
    # med_err = np.median(data['median_err'], axis=0)[1:]
    if node_err:
        med_err = np.median(data['node_err'], axis=0)
        mat_file = 'noise_node_err'
    else:
        med_err = np.median(data['median_err'], axis=0)
        mat_file = 'noise'
    xlabel = 'Normalized noise power'

    # legend = ['Enc/Dec Wei(192)', 'AutoConv (210)',
    #           'AutoFC (193)', 'Enc/Dec Wei(132)', 'Enc/Dec Wei(102)',
    #           'AutoConv (140)']
    # fmt = ['o-', 'P-', 'X-', 'o--', 'o:', 'P--']

    legend = ['AE-FC-709', 'G-E/D-440', 'AE-CV-429', 'G-E/D-298', 'AE-CV-308',
              'G-E/D-132', 'AE-CV-144']
    fmt = ['X-', 'o-', 'P-', 'o--', 'P--', 'o:', 'P:']

    plot_results(noise, med_err, xlabel=xlabel, fmt=fmt, legend=legend)
    save_mat_file('mat_file', med_err, noise, legend, fmt)


def plot_node_pert(file_name, node_err=True):
    path = PATH + 'node_pert/' + file_name + '.npy'
    data = np.load(path).item()
    pct = np.array([10, 20, 30, 40, 50])
    print((256-pct))
    if node_err:
        med_err = np.median(data['node_err'], axis=0)
        mat_file = 'nodes_pert_node_err'
    else:
        med_err = np.median(data['median_err'], axis=0)
        mat_file = 'nodes_pert'
    xlabel = 'Number of deleted nodes'
    legend = ['AE-FC-709', 'G-E/D-440', 'G-E/D-298', 'AE-CV-308',
              'G-E/D-132', 'AE-CV-144']
    fmt = ['X-', 'o-', 'o--', 'P--', 'o:', 'P:']

    # med_err = med_err[np.array([0, 2, 3, 4, 5])]
    legend.remove(legend[1])
    fmt.remove(fmt[1])

    plot_results(pct, med_err, xlabel=xlabel, fmt=fmt, legend=legend)
    save_mat_file(mat_file, med_err, pct, legend, fmt)


def plot_deltas(file_name, node_err=True):
    path = PATH + 'deltas/' + file_name + '.npy'
    data = np.load(path).item()
    deltas = [4, 40, 80, 120, 160, 200]
    if node_err:
        med_err = np.median(data['node_err'], axis=0)
        mat_file = 'deltas_node_err'
    else:
        med_err = np.median(data['median_err'], axis=0)
        mat_file = 'deltas'
    xlabel = 'Number of deltas'
    legend = ['AE-FC-709', 'G-E/D-298', 'AE-CV-308',
              'G-E/D-132', 'AE-CV-144']
    fmt = ['X-', 'o--', 'P--', 'o:', 'P:']
    plot_results(deltas, med_err, xlabel=xlabel, fmt=fmt, legend=legend)
    save_mat_file(mat_file, med_err, deltas, legend, fmt)


if __name__ == '__main__':
    # plot_perturbation('perturbation_2019_10_24-20_23', False)
    # plot_noise('noise_2019_10_27-00_33', False)
    # plot_node_pert('node_pert_2019_10_27-04_05', False)
    plot_deltas('deltas_2019_10_27-12_10', True)

    # path = PATH + 'deltas/' + 'deltas_2019_10_27-12_10.npy'
    # data = np.load(path).item()
    # print_results(data['signals']['deltas'], data['exps'],
    #               data['node_err'], data['median_err'])