import torch
import torch.optim as optim
import numpy as np
import time
from tensorboardX import SummaryWriter
import copy

VERB = True
DEBUG = False
DEEP_DEBUG = False

class Model:
    def __init__(self,
                arch,
                optimizer, learning_rate, beta1, beta2, decay_rate, loss_func,
                num_epochs, batch_size, eval_freq, max_non_dec,
                tb_log):
        # Define architecture
        self.arch = arch

        # Define optimizer
        if optimizer == 'ADAM':
            self.optim = optim.Adam(self.arch.parameters(), lr=learning_rate, betas=(beta1, beta2))
        else:
            self.optim = optim.SGD(self.arch.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optim, decay_rate)

        self.loss_func = loss_func
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.max_non_dec = max_non_dec
        self.tb_log = tb_log

        if DEEP_DEBUG:
            for p in self.arch.parameters():
                p.register_hook(lambda grad: print(grad))

    def fit(self, data, labels, val_data, val_labels):
        """
        Train the model
        """
        n_samples = data.shape[0]

        num_steps = int(n_samples / self.batch_size)

        t_init = time.time()
        t_step = t_init

        best_err = 10000000
        cont = 0

        for i in range(1, self.num_epochs+1):
            for j in range(1, num_steps + 1):
                idx = np.random.permutation(n_samples)[0:self.batch_size]

                batch_data = data[idx,:]
                batch_labels = labels[idx]

                # Reset gradients
                self.arch.zero_grad()

                self.train(batch_data, batch_labels)

            self.scheduler.step()

            # Samuel's early stopping
            if self.max_non_dec != None:
                loss, acc, mean_err = self.predict(val_data, val_labels, i)
                if loss.data*1.005 < best_err:
                    best_err = loss.data
                    best_net = copy.deepcopy(self.arch)
                    cont = 0
                else:
                    if cont >= self.max_non_dec:
                        if VERB:
                            print("Early Stopping at {} epochs.".format(str(i)))
                        break
                    cont += 1

            if i % self.eval_freq == 0 and VERB:
                loss, acc, mean_err = self.predict(val_data, val_labels, i)
                print('Epoch {}/{}'.format(i, self.num_epochs), end=" - ")
                if self.class_type:
                    print('Accuracy: {} ({}/{})'.format(round(acc * 100.0, 2), int(round(acc*val_data.shape[0])), val_data.shape[0]), end=" - ")
                print('Loss: {:.8f}'.format(loss), end=" - ")
                if not self.class_type:
                    print('Mean Err: {:.8f}'.format(mean_err), end=" - ")
                now = time.time()
                print('Time: {} (step) - {} (since beginning)'.format(round(now-t_step, 2), round(now-t_init,2)))
                t_step = now

                if self.tb_log:
                    self.writer.add_scalar('perf/accuracy', acc, i)
                    self.writer.add_scalar('perf/loss', loss.item(), i)

        # Taking the best architecture from early stopping
        if self.max_non_dec != None:
            self.arch = best_net
        loss, acc, mean_err = self.predict(data, labels)
        if VERB:
            if self.class_type:
                print('Training Accuracy: {} ({}/{})'.format(round(acc * 100.0, 2), int(round(acc*data.shape[0])), data.shape[0]), end=" - ")
            print('Training Loss: {}'.format(loss), end=" - ")
            print('Training Mean Err: {:.8f}'.format(mean_err))

    def predict(self, data, labels, it=None):
        """
        Predicts using the architecture
        Returns loss and accuracy
        """
        with torch.no_grad():
            y_pred = self.arch(data)
            if DEBUG:
                print("Data: " + str(data))
                print("Predictions: " + str(y_pred))
                print("Labels: " + str(labels))
                print("Y_pred shape:" + str(y_pred.shape))
            if self.tb_log and it != None:
                self.writer.add_histogram('pred/y_pred', y_pred, it)
                c = 0
                for p in self.arch.parameters():
                    self.writer.add_histogram('pred/' + self.arch.l_param[c], p.data, it)
                    c += 1

            mse_loss = self.loss_func(y_pred, labels)

            if self.class_type:
                predictions, index = y_pred.max(1) # Max along dimension 1. Removes dimension 1 (cols)
                if DEBUG:
                    print("Index: " + str(index))
                    print("Predictions: " + str(predictions))

                # Item function returns the scalar in a 1 element tensor
                num_ok = torch.sum(torch.eq(index, labels)).item()
                acc = float(num_ok / data.shape[0])
                mean_norm_error = 0.0
            else:
                acc = 0.0
                # Samuel mean norm error
                y_pred_np = y_pred.detach().numpy()
                labels_np = labels.detach().numpy()
                #print(y_pred_np, labels_np)
                norm_error = np.sum((y_pred_np-labels_np)**2,axis=1)/np.linalg.norm(labels_np,axis=1)
                mean_norm_error = np.mean(norm_error)

        return mse_loss, acc, mean_norm_error

    def eval(self, train_data, train_labels, val_data, val_labels, test_data, test_labels):
        if self.tb_log:
            self.writer = SummaryWriter()
            #self.writer.add_graph(self.arch)

        self.class_type = len(train_labels.shape) == 1    # If labels just have the
        # train data dimension, it is a classification problem

        # Turn data into tensors
        train_data = torch.FloatTensor(train_data)
        test_data = torch.FloatTensor(test_data)
        if self.class_type:
            train_labels = torch.LongTensor(train_labels)
            val_labels = torch.LongTensor(val_labels)
            test_labels = torch.LongTensor(test_labels)
        else:
            train_labels = torch.FloatTensor(train_labels)
            val_labels = torch.FloatTensor(val_labels)
            test_labels = torch.FloatTensor(test_labels)

        self.fit(train_data, train_labels, val_data, val_labels)
        if self.tb_log:
            self.writer.close()
        return self.predict(test_data, test_labels)


    def train(self, data, labels):
        # Run the architecture
        logits = self.arch(data)

        if DEEP_DEBUG:
            print('Logits')
            print(logits)
            print(logits.shape)
            print('labels')
            print(labels)
            print(labels.shape)
        loss = self.loss_func(logits, labels)

        # print(loss.grad_fn)
        #print(loss)
        loss.backward()
        if DEEP_DEBUG:
            for p in self.arch.parameters():
                print(p)
        self.optim.step()
