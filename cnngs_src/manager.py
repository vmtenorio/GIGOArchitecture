"""
Code below has been kindly provided by Michäel Defferrard
http://github.com/mdeff/cnn_graph

Minor modifications have been made.
"""

import sklearn
import sklearn.grid_search
import matplotlib.pyplot as plt
import numpy as np

def grid_search(params, grid_params, train_data, train_labels, val_data,
		val_labels, test_data, test_labels, model):
	"""Explore the hyper-parameter space with an exhaustive grid search."""
	params = params.copy()
	train_accuracy, test_accuracy, train_f1, test_f1 = [], [], [], []
	grid = sklearn.grid_search.ParameterGrid(grid_params)
	print('grid search: {} combinations to evaluate'.format(len(grid)))
	for grid_params in grid:
		params.update(grid_params)
		name = '{}'.format(grid)
		print('\n\n  {}  \n\n'.format(grid_params))
		m = model(params)
		m.fit(train_data, train_labels, val_data, val_labels)
		string, accuracy, f1, loss = m.evaluate(train_data, train_labels)
		train_accuracy.append('{:5.2f}'.format(accuracy)); train_f1.append('{:5.2f}'.format(f1))
		print('train {}'.format(string))
		string, accuracy, f1, loss = m.evaluate(test_data, test_labels)
		test_accuracy.append('{:5.2f}'.format(accuracy)); test_f1.append('{:5.2f}'.format(f1))
		print('test  {}'.format(string))
	print('\n\n')
	print('Train accuracy:      {}'.format(' '.join(train_accuracy)))
	print('Test accuracy:       {}'.format(' '.join(test_accuracy)))
	print('Train F1 (weighted): {}'.format(' '.join(train_f1)))
	print('Test F1 (weighted):  {}'.format(' '.join(test_f1)))
	for i,grid_params in enumerate(grid):
		print('{} --> {} {} {} {}'.format(
				grid_params, train_accuracy[i], test_accuracy[i],
				train_f1[i], test_f1[i]))

class model_manager(object):

	def __init__(s):
		s.names, s.params = set(), {}
		s.fit_accuracies, s.fit_losses, s.fit_time = {}, {}, {}
		s.train_accuracy, s.train_f1, s.train_loss = {}, {}, {}
		s.test_accuracy, s.test_f1, s.test_loss = {}, {}, {}
		s.n_param = {}
		s.acc_graph_mod = {}
		s.test_accuracy_mod, s.test_f1_mod, s.test_loss_mod = {}, {}, {}

	def test(s, model, name, params, train_data, train_labels, val_data,
			val_labels, test_data, test_labels):
		s.params[name] = params
		s.n_param[name] = model.n_param
		s.fit_accuracies[name], s.fit_losses[name], s.fit_time[name] = \
				model.fit(train_data, train_labels, val_data, val_labels)
		string, s.train_accuracy[name], s.train_f1[name], s.train_loss[name] = \
				model.evaluate(train_data, train_labels)
		print('train {}'.format(string))
		string, s.test_accuracy[name], s.test_f1[name], s.test_loss[name] = \
				model.evaluate(test_data, test_labels)
		print('test  {}'.format(string))
		s.names.add(name)
		
	def evaluate_mod(s, model, name, test_data, test_labels):
		string, s.test_accuracy_mod[name], s.test_f1_mod[name], s.test_loss_mod[name] = \
				model.evaluate(test_data, test_labels)
		print('test  graph mod {}'.format(string))

	def show(s, fontsize=None):
		if fontsize:
			plt.rc('pdf', fonttype=42)
			plt.rc('ps', fonttype=42)
			plt.rc('font', size=fontsize)     #controls default text sizes
			plt.rc('axes', titlesize=fontsize)#fontsize of the axes title
			plt.rc('axes', labelsize=fontsize)#fontsize of the x and y labels
			plt.rc('xtick', labelsize=fontsize)#fontsize of the tick labels
			plt.rc('ytick', labelsize=fontsize)#fontsize of the tick labels
			plt.rc('legend', fontsize=fontsize)#legend fontsize
			plt.rc('figure', titlesize=fontsize)#size of the figure title
		#!!! fgama 2017/10/18 Changed the way results are shown to illustrate
		# the difference in the number of parameters

		for name in sorted(s.names):
			print("    {} = {{F = {}, K = {}, M = {}}}".
						format(name,s.params[name]['F'],
						s.params[name]['K'], s.params[name]['M']))
		print(" ")
		print("    Results:")
		print('      accuracy        F1        parameters    time [ms]  name')
		print('    test  train   test  train   ')
		for name in sorted(s.names):
			print('    {:5.2f} {:5.2f}   {:5.2f} {:5.2f}   {:7.0f}        {:3.0f}       {}'.format(
					s.test_accuracy[name], s.train_accuracy[name],
					s.test_f1[name], s.train_f1[name],
					s.n_param[name], s.fit_time[name]*1000, name))
		#!!!

		fig, ax = plt.subplots(1, 2, figsize=(15, 5))
		for name in sorted(s.names):
			steps = np.arange(len(s.fit_accuracies[name])) + 1
			steps *= s.params[name]['eval_frequency']
			ax[0].plot(steps, s.fit_accuracies[name], '.-', label=name)
			ax[1].plot(steps, s.fit_losses[name], '.-', label=name)
		ax[0].set_xlim(min(steps), max(steps))
		ax[1].set_xlim(min(steps), max(steps))
		ax[0].set_xlabel('step')
		ax[1].set_xlabel('step')
		ax[0].set_ylabel('validation accuracy')
		ax[1].set_ylabel('training loss')
		ax[0].legend(loc='lower right')
		ax[1].legend(loc='upper right')
		#fig.savefig('training.pdf')
