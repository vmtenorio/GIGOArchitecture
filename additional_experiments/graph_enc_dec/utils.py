import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from scipy.io import savemat


PATH = './results/'


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


def plot_perturbation(file_name, node_err=True, save=False):
    path = PATH + 'perturbation/' + file_name + '.npy'
    data = np.load(path).item()
    med_err = np.median(data['median_err'], axis=0)
    xlabel = '% of perturbed links'

    if node_err:
        med_err = np.median(data['node_err'], axis=0)
        mat_file = 'perturbation_node_err'
    else:
        med_err = np.median(data['median_err'], axis=0)
        mat_file = 'perturbation'

    # Last experiments
    pct = [val[0] + val[1] for val in data['Gs']['pct_val']]
    fmt = ['X-', 'o-', '^-', 'P-', 'o--', 'P--', 'o:', '^:', 'P:']
    legend = ['AE-FC-769', 'G-E/D-Wei-440', 'G-E/D-NoUps-440', 'AE-CV-440',
              'G-E/D-Wei-298', 'AE-CV-308',
              'G-E/D-Wei-132', 'G-E/D-NoUps-132', 'AE-CV-144']

    for i, exp in enumerate(data['exps']):
        print(i, exp)
        print('err:', med_err[i, :])
    # med_err = med_err[np.array([0, 1, 2, 3, 5, 6, 8]), :]
    legend.remove(legend[7])
    # legend.remove(legend[5])
    legend.remove(legend[2])
    fmt.remove(fmt[7])
    # fmt.remove(fmt[5])
    fmt.remove(fmt[2])

    plot_results(pct, med_err, xlabel=xlabel, fmt=fmt, legend=legend)
    if save:
        save_mat_file(mat_file, med_err, pct, legend, fmt)


def plot_noise(file_name, node_err=True, save=False):
    path = PATH + 'noise/' + file_name + '.npy'
    data = np.load(path).item()
    noise = data['signals']['noise']
    if node_err:
        med_err = np.median(data['node_err'], axis=0)
        mat_file = 'noise_node_err'
    else:
        med_err = np.median(data['median_err'], axis=0)
        mat_file = 'noise'
    xlabel = 'Normalized noise power'
    # legend = ['AE-FC-709', 'G-E/D-440', 'AE-CV-429', 'G-E/D-298', 'AE-CV-308',
    #           'G-E/D-132', 'AE-CV-144']
    # fmt = ['X-', 'o-', 'P-', 'o--', 'P--', 'o:', 'P:']
    legend = ['AE-FC-769', 'G-E/D-Wei-762', 'G-E/D-None-762', 'AE-CV-792',
              'G-E/D-Wei-132', 'G-E/D-None-132', 'AE-CV-144']
    fmt = ['X-', 'o-', '^-', 'P-', 'o--', '^--', 'P--']
    plot_results(noise, med_err, xlabel=xlabel, fmt=fmt, legend=legend)
    if save:
        save_mat_file(mat_file, med_err, noise, legend, fmt)


def plot_noise2(file_name, node_err=True, save=False):
    path = PATH + 'noise/' + file_name + '.npy'
    data = np.load(path).item()
    noise = data['signals']['noise']
    if node_err:
        med_err = np.median(data['node_err'], axis=0)
        mat_file = 'noise_node_err2'
    else:
        med_err = np.median(data['median_err'], axis=0)
        mat_file = 'noise2'
    xlabel = 'Normalized noise power'
    legend = ['AE-FC-2158', 'G-E/D-2225', 'AE-CV-2145',
              'AE-FC-1192', 'G-E/D-1180', 'AE-CV-1188',
              'AE-FC-709', 'G-E/D-706', 'AE-CV-693']
    fmt = ['X-', 'o-', 'P-', 'X--', 'o--', 'P--', 'X:', 'o:', 'P:',]
    plot_results(noise, med_err, xlabel=xlabel, fmt=fmt, legend=legend)
    if save:
        save_mat_file(mat_file, med_err, noise, legend, fmt)


def plot_node_pert(file_name, node_err=True, save=False):
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
              'G-E/D-132', 'AE-CV-132']
    fmt = ['X-', 'o-', 'o--', 'P--', 'o:', 'P:']

    # med_err = med_err[np.array([0, 2, 3, 4, 5])]
    legend.remove(legend[1])
    fmt.remove(fmt[1])

    plot_results(pct, med_err, xlabel=xlabel, fmt=fmt, legend=legend)
    if save:
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


def plot_samples(file_name, node_err=True):
    path = PATH + 'samples/' + file_name + '.npy'
    data = np.load(path).item()
    samples = data['signals']['samples']
    if node_err:
        med_err = np.median(data['node_err'], axis=0)
        mat_file = 'samples_node_err'
    else:
        med_err = np.median(data['median_err'], axis=0)
        mat_file = 'samples'
    xlabel = 'Number of deltas'
    legend = ['AE-FC-709', 'G-E/D-762', 'AE-CV-792',
              'G-E/D-298', 'AE-CV-286', 'G-E/D-132', 'AE-CV-144']
    fmt = ['X-', 'o-', 'P-', 'o--', 'P--', 'o:', 'P:']

    med_err = med_err[np.array([0, 1, 2, 4, 5, 6]), :]
    legend.remove(legend[3])
    # legend.remove(legend[3])
    fmt.remove(fmt[3])
    # fmt.remove(fmt[3])

    plot_results(samples, med_err, xlabel=xlabel, fmt=fmt, legend=legend)
    save_mat_file(mat_file, med_err, samples, legend, fmt)


if __name__ == '__main__':
    # path = PATH + 'perturbation/' + 'perturbation_2019_10_30-04_46.npy'
    # data = np.load(path).item()
    # print_results(data['Gs']['pct_val'], data['exps'],
    #               data['node_err'], data['median_err'])

    plot_perturbation('perturbation_2019_10_30-04_46', True, True)
    # plot_noise('noise_2019_10_27-10_39', True, True)
    # plot_node_pert('node_pert_2019_10_27-04_05', True, True)
    # plot_deltas('deltas_2019_10_27-12_10', True)
    # plot_samples('samples_2019_10_29-09_37', True)
