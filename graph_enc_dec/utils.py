import numpy as np
import os
import datetime


# NOTE: maybe create a class for print/plot?
def print_partial_results(p_n, exps, mean_err, med_err):
    print('Noise:', p_n)
    for i, exp in enumerate(exps):
        print('{}. {}\n\tMean Err: {} STD: {}\n\tMedian Err: {} STD: {}'
              .format(i+1, exp, np.median(mean_err[:, i]), np.std(mean_err[:, i]),
                      np.median(med_err[:, i]), np.std(med_err[:, i])))


def print_results(P_n, exps, mean_err, med_err):
    for i, p_n in enumerate(P_n):
        print_partial_results(p_n, exps, mean_err[:, :, i], med_err[:, :, i])
        print()


def save(path, exps, mean_err, med_err, Gs, signals, learning):
    if not os.path.isdir(path):
        os.makedirs(path)

    data = {}
    data['Gs'] = Gs
    data['signals'] = signals
    data['exps'] = exps
    data['mean_err'] = mean_err
    data['median_err'] = med_err
    data['learning'] = learning
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
    file_name = path.split('/')[-1] + '_{}'.format(timestamp)
    file_path = path + '/' + file_name
    np.save(file_path, data)
    print('Saved as:', file_path)


def plot_results(err):
    return


def load_results(path):
    return
