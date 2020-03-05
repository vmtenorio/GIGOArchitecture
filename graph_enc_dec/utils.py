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


def plot_results(err, x_axis, legend=None, fmts=None, x_label=None):
    median_err = np.median(err, axis=1)
    plt.subplots()
    for i in range(median_err.shape[1]):
        if fmts is None:
            plt.semilogy(x_axis, median_err[:, i])
        else:
            plt.semilogy(x_axis, median_err[:, i], fmts[i])

    if x_label is None:
        x_label = 'Normalized Noise Power'
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel('Median Error', fontsize=16)
    plt.gca().autoscale(enable=True, axis='x', tight=True)
    plt.grid(True, which='both')
    if legend is not None:
        plt.legend(legend)
        # plt.legend(legend, prop={'size': 14})
    plt.show()


def print_partial_results(p_n, exps, node_err, med_err):
    print('Noise:', p_n)
    for i, exp in enumerate(exps):
        print('{}. {}\n\tNode Err: {} STD: {}\n\tMedian Err: {} STD: {}'
              .format(i+1, exp, np.median(node_err[:, i]), np.std(node_err[:, i]),
                      np.median(med_err[:, i]), np.std(med_err[:, i])))


def print_results(P_n, exps, node_err, med_err):
    for i, p_n in enumerate(P_n):
        print_partial_results(p_n, exps, node_err[i, :, :], med_err[i, :, :])
        print()


def plot_from_file(file, skip=[]):
    data = np.load(file).item()
    if skip:
        data = remove_indexes(data, skip)

    err = data['err']
    if 'Pert' in data:
        noise = np.array(data['Pert'])
        if len(noise.shape) > 1:
            noise = noise.sum(axis=1)
    else:
        noise = np.arange(err.shape[0])
    legend = data['legend']
    if 'x_label' in data:
        xlabel = data['x_label']
    else:
        xlabel = None
    fmts = data['fmts']
    # legend = data['legend']
    # fmts = data['fmts']

    # if isinstance(noise, list):
    #     plot_results(err, noise, legend, fmts)
    # else:
    #     p_miss = data['Signals']['P_MISS']
    #     x_label = 'Percentage of missing values'
    plot_results(err, noise, legend=legend, fmts=fmts, x_label=xlabel)


def print_sumary(data):
    print('Graph parameters:')
    print(data['Gs'])
    print('Signals parameters:')
    print(data['Signals'])
    print('Network parameters:')
    print(data['Net'])
    print('Experiments:')
    print(data['exps'])


def print_from_file(file, skip=[]):
    data = np.load(file).item()
    if skip:
        data = remove_indexes(data, skip)
    err = data['err']
    node_err = data['node_err']
    noise = np.arange(err.shape[0])
    exps = data['exps']
    print_sumary(data)
    print_results(noise, exps, node_err, err)


def remove_indexes(data, skip_indexes):
    data['err'] = np.delete(data['err'], skip_indexes, axis=2)
    skip_indexes.sort(reverse=True)
    for i in skip_indexes:
        del data['exps'][i]
        del data['legend'][i]
        del data['fmts'][i]
    return data


def save_mat_file(file, skip=[]):
    data = np.load(file).item()
    if skip:
        data = remove_indexes(data, skip)

    exp_type = file.split('/')[-2]
    fmts = []
    for fmt in data['fmts']:
        fmts.append(fmt.replace('P', '+'))

    mat_file = './results/matfiles/' + exp_type
    mat_data = {'error': data['err'], 'leg': data['legend'],
                'fmts': fmts}
    savemat(mat_file, mat_data)
    print('Saved as:', mat_file)


if __name__ == '__main__':
    skiped = []
    file_name = 'results/nodes_pert/nodes_2020_02_29-23_50' + '.npy'
    # print_from_file(file_name, skip=skiped)
    plot_from_file(file_name, skip=skiped)
    save_mat_file(file_name, skip=skiped)

    # path = PATH + 'deltas/' + 'deltas_2019_10_27-12_10.npy'
    # data = np.load(path).item()
    # print_results(data['signals']['deltas'], data['exps'],
    #               data['node_err'], data['median_err'])
