from torch import optim, no_grad, nn
import copy
import numpy as np
import sys


class Model:
    # TODO: add support for more optimizers
    def __init__(self, arch,
                 learning_rate=0.1, decay_rate=0.99, loss_func=nn.MSELoss(),
                 epochs=50, batch_size=100, eval_freq=5, verbose=False,
                 max_non_dec=10):
        self.arch = arch
        # self.lr = learning_rate
        # self.decay = decay_rate
        self.loss = loss_func
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.verbose = verbose
        self.max_non_dec = max_non_dec
        self.optim = optim.Adam(self.arch.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optim, decay_rate)

    def count_params(self):
        return sum(p.numel() for p in self.arch.parameters() if p.requires_grad)

    def fit(self, train_X, train_Y, val_X, val_Y):
        n_samples = train_X.shape[0]
        n_steps = int(n_samples/self.batch_size)

        best_err = 1000000
        best_net = None
        cont = 0
        for i in range(1, self.epochs+1):
            for j in range(n_steps):
                # Randomly seect batches
                idx = np.random.permutation(n_samples)[:self.batch_size]
                batch_X = train_X[idx,:]
                batch_Y = train_Y[idx,:]
                self.arch.zero_grad()

                # Training step
                predicted_Y = self.arch(batch_X)
                training_loss = self.loss(predicted_Y, batch_Y)
                training_loss.backward()
                self.optim.step()

            self.scheduler.step()
            # Predict eval error
            with no_grad():
                predicted_Y_eval = self.arch(val_X)
                eval_loss = self.loss(predicted_Y_eval, val_Y)
            if eval_loss.data*1.005 < best_err:
                best_err = eval_loss.data
                best_net = copy.deepcopy(self.arch)
                cont = 0
            else:
                if cont >= self.max_non_dec:
                    break
                cont += 1

            if self.verbose and i % self.eval_freq == 0:
                print('Epoch {}/{}: \tEval loss: {:.8f} \tTrain loss: {:.8f}'
                    .format(i, self.epochs, eval_loss, training_loss))
        self.arch = best_net

    def test(self, test_X, test_Y):
        # Ignoring dim[1] with only one channel
        shape = [test_X.shape[0], test_X.shape[2]]

        # Error for each node
        test_Y = test_Y.view(shape)
        Y_hat = self.arch(test_X).view(shape)
        node_mse = self.loss(Y_hat, test_Y)

        # Normalize error for the whole signal
        Y_hat = Y_hat.detach().numpy()
        test_Y = test_Y.detach().numpy()
        norm_error = np.sum((Y_hat-test_Y)**2, axis=1)/np.linalg.norm(test_Y, axis=1)
        mean_norm_error = np.mean(norm_error)
        median_norm_error = np.median(norm_error)
        return mean_norm_error, median_norm_error, node_mse.detach().numpy()

"""
class LinearModel(Model):
    def __init__(self, N, loss_func=nn.MSELoss(), verbose=False):
        self.arch = LinearReg(N)
        self.N = N
        self.loss = loss_func
        self.verbose = verbose

    def count_params(self):
        return self.N**2

    def fit(self, train_X, train_Y):
        X = train_X.view([train_X.shape[0], train_X.shape[2]])
        Y = train_Y.view([train_Y.shape[0], train_Y.shape[2]])
        X2 = X.t().mm(X)
        self.arch.Beta = X2.pinverse().mm(X.t()).mm(Y)

        if self.verbose:
            Y_est = X.mm(self.arch.Beta)
            print(Y_est.shape)
            print(X.shape)
            print(self.arch.Beta.shape)
            err = np.sum((Y_est-Y)**2, axis=1)/np.linalg.norm(Y)**2
            print('Train Error:', np.mean(err), np.median(err))
"""
