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
    legend = ['AutoFC (769)', 'Enc/Dec Wei(440)', 'AutoConv (440)',
              'Enc/Dec Wei(298)', 'Enc/Dec NoUps(298)', 'AutoConv (280)',
              'Enc/Dec Wei(132)', 'Enc/Dec NoUps(132)', 'AutoConv (140)']

    plot_results(pct, err, xlabel=xlabel, fmt=fmt, legend=legend)
    save_mat_file('perturbation', err, pct, legend, fmt)


def plot_noise(file_name):
    path = PATH + 'diff_models/' + file_name + '.npy'
    data = np.load(path).item()
    noise = data['signals']['noise']
    # med_err = np.median(data['median_err'], axis=0)[1:]
    med_err = np.median(data['median_err'], axis=0)
    xlabel = 'Normalized noise power'

    # legend = ['Enc/Dec Wei(192)', 'AutoConv (210)',
    #           'AutoFC (193)', 'Enc/Dec Wei(132)', 'Enc/Dec Wei(102)',
    #           'AutoConv (140)']
    # fmt = ['o-', 'P-', 'X-', 'o--', 'o:', 'P--']

    legend = ['AutoencFC(709)', 'Enc/Dec WEI (440)', 'AutoencConv (429)',
              'Enc/Dec WEI (298)', 'AutoencConv (308)', 'Enc/Dec Wei(132)',
              'AutoConv (144)']
    fmt = ['X-', 'o-', 'P-', 'o--', 'P--', 'o:', 'P:']

    plot_results(noise, med_err, xlabel=xlabel, fmt=fmt, legend=legend)
    save_mat_file('noise2', med_err, noise, legend, fmt)


def plot_node_pert(file_name):
    path = PATH + 'node_pert/' + file_name + '.npy'
    data = np.load(path).item()
    pct = [10, 20, 30, 40, 50]
    med_err = np.median(data['median_err'], axis=0)
    xlabel = 'Number of deleted nodes'
    legend = ['AutoencFC (709)', 'Enc/Dec (440)', 'Enc/Dec (298)', 'AutoencConv (308)',
              'Enc/Dec (132)', 'AutoencConv (143)']
    fmt = ['X-', 'o-', 'o--', 'P--', 'o:', 'P:']
    plot_results(pct, med_err, xlabel=xlabel, fmt=fmt, legend=legend)
    save_mat_file('nodes_pert', med_err, pct, legend, fmt)


if __name__ == '__main__':
    plot_perturbation('perturbation_2019_10_24-20_23', False)
    # plot_noise('diff_models_2019_10_24-22_04')
    # plot_node_pert('node_pert_2019_10_25-12_38')
