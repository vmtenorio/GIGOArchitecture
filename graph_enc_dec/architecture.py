from torch import manual_seed, nn, Tensor, optim, no_grad
import copy
import torch
import sys
import numpy as np
import time

from graph_enc_dec.graph_clustering import NONE, REG, NO_A, BIN, WEI


# NOTE: maybe is good to pass method as argument...
class GraphEncoderDecoder(nn.Module):
    def __init__(self,
                 # Encoder args
                 features_enc, nodes_enc, Ds,
                 # Decoder args
                 features_dec, nodes_dec, Us,
                 # Only conv layers
                 features_conv_dec=[],
                 # Optional args
                 As_enc=None, As_dec=None, ups=WEI, downs=WEI,
                 gamma=0.5, batch_norm=True,
                 # Activation functions
                 act_fn=nn.Tanh(), last_act_fn=nn.Tanh()):
        if features_enc[-1] != features_dec[0] or nodes_enc[-1] != nodes_dec[0]:
            raise RuntimeError('Different dimensions for the latent space')

        if len(features_conv_dec) != 0 and features_conv_dec[0] != features_dec[-1]:
            raise RuntimeError('Features of last decoder layer and first conv layer do not match')

        if len(features_enc) != len(nodes_enc) or len(features_dec) != len(nodes_dec):
            raise RuntimeError('Length of the nodes and features vector must be the same')

        super(GraphEncoderDecoder, self).__init__()
        self.model = nn.Sequential()
        self.fts_enc = features_enc
        self.nodes_enc = nodes_enc
        self.Ds = Ds
        self.fts_dec = features_dec
        self.nodes_dec = nodes_dec
        self.Us = Us
        self.fts_cnv_dec = features_conv_dec
        self.As_enc = As_enc
        self.As_dec = As_dec
        self.ups = ups
        self.downs = downs
        self.kernel_size = 1
        self.gamma = gamma
        self.batch_norm = batch_norm
        self.act_fn = act_fn
        self.last_act_fn = last_act_fn
        self.build_network()

    def add_layer(self, module):
        self.model.add_module(str(len(self.model) + 1), module)

    # TODO: create a method for testing the correct creation of the arch
    # NOTE: possibility for a section of only convolutions at the beggining
    def build_network(self):
        # Encoder Section
        downs_skip = 0
        for l in range(len(self.fts_enc)-1):
            self.add_layer(nn.Conv1d(self.fts_enc[l], self.fts_enc[l+1],
                           self.kernel_size, bias=False))

            if self.nodes_enc[l] > self.nodes_enc[l+1]:
                if self.As_enc:
                    A = self.As_enc[l+1-downs_skip]
                else:
                    A = None
                self.add_layer(GraphDownsampling(self.Ds[l-downs_skip],
                                                 A, self.gamma, self.downs))
            else:
                downs_skip += 1
            if self.act_fn is not None:
                self.add_layer(self.act_fn)
            if self.batch_norm:
                self.add_layer(nn.BatchNorm1d(self.fts_enc[l+1]))

        # Decoder Section
        ups_skip = 0
        for l in range(len(self.fts_dec)-1):
            self.add_layer(nn.Conv1d(self.fts_dec[l], self.fts_dec[l+1],
                           self.kernel_size, bias=False))

            if self.nodes_dec[l] < self.nodes_dec[l+1]:
                if self.As_dec:
                    A = self.As_dec[l+1-ups_skip]
                else:
                    A = None
                self.add_layer(GraphUpsampling(self.Us[l-ups_skip],
                                               A, self.gamma, self.ups))
            else:
                ups_skip += 1

            if len(self.fts_cnv_dec) > 0 or l < (len(self.fts_dec)-2):
                # This is not the last layer
                if self.act_fn is not None:
                    self.add_layer(self.act_fn)
                if self.batch_norm:
                    self.add_layer(nn.BatchNorm1d(self.fts_dec[l+1]))
            else:
                # This is the last layer
                if self.last_act_fn is not None:
                    self.add_layer(self.last_act_fn)

        # Only convolutions section
        for l in range(len(self.fts_cnv_dec)-1):
            self.add_layer(nn.Conv1d(self.fts_cnv_dec[l], self.fts_cnv_dec[l+1],
                           self.kernel_size, bias=False))
            if l < len(self.fts_cnv_dec)-2:
                if self.act_fn is not None:
                    self.add_layer(self.act_fn)
                if self.batch_norm:
                    self.add_layer(nn.BatchNorm1d(self.fts_cnv_dec[l+1]))
            else:
                # Last layer
                if self.last_act_fn is not None:
                    self.add_layer(self.last_act_fn)

    def forward(self, x):
        # TODO: Separate in encoder, decoder and conv (?)
        return self.model(x)


# TODO: common class GraphSampling(?)
class GraphUpsampling(nn.Module):
    """
    Use information from the agglomerative hierarchical clustering for
    doing the upsampling by creating the upsampling matrix U
    """
    def __init__(self, U, A, gamma=0.5, method=WEI):  # WEI
        # NOTE: gamma = 1 is equivalent to no_A
        super(GraphUpsampling, self).__init__()
        if A is not None:
            assert np.allclose(A, A.T), 'A should be symmetric'
            self.A = np.linalg.inv(np.diag(np.sum(A, 0))).dot(A)
            if method in [BIN, WEI]:
                self.A = gamma*np.eye(A.shape[0]) + (1-gamma)*self.A
            self.A = Tensor(self.A)

        self.parent_size = U.shape[1]
        self.child_size = U.shape[0]
        self.method = method
        self.U_T = Tensor(U).t()

    def iterate_over_channs(self, input, output, n_channels):
        for i in range(n_channels):
            if self.method == NO_A:
                output[:, i, :] = input[:, i, :].mm(self.U_T)
            elif self.method in [BIN, WEI]:
                output[:, i, :] = input[:, i, :].mm(self.U_T).mm(self.A)
            else:
                raise RuntimeError('Unknown sampling method')
        return output

    def iterate_over_samples(self, input, output, n_samples):
        for i in range(n_samples):
            if self.method == NO_A:
                output[i, :, :] = input[i, :, :].mm(self.U_T)
            elif self.method in [BIN, WEI]:
                output[i, :, :] = input[i, :, :].mm(self.U_T).mm(self.A)
            else:
                raise RuntimeError('Unknown sampling method')
        return output

    def forward(self, input):
        if self.method == REG:
            sf = self.child_size/self.parent_size
            return nn.functional.interpolate(input, scale_factor=sf,
                                             mode='linear',
                                             align_corners=True)
        n_samples = input.shape[0]
        n_channels = input.shape[1]
        output = torch.zeros([n_samples, n_channels, self.child_size])
        if n_channels < n_samples:
            return self.iterate_over_channs(input, output, n_channels)
        else:
            return self.iterate_over_samples(input, output, n_samples)


class GraphDownsampling(nn.Module):
    def __init__(self, D, A, gamma=0.5, method=WEI):
        super(GraphDownsampling, self).__init__()
        if A is not None:
            assert np.allclose(A, A.T), 'A should be symmetric'
            self.A = np.linalg.inv(np.diag(np.sum(A, 0))).dot(A)
            if method in [BIN, WEI]:
                self.A = gamma*np.eye(A.shape[0]) + (1-gamma)*self.A
            self.A = Tensor(self.A)

        self.method = method
        self.parent_size = D.shape[1]
        self.child_size = D.shape[0]
        # NOTE: only creation of D changes!
        Deg_inv = np.linalg.inv(np.diag(np.sum(D, 1)))
        self.D_T = Tensor(Deg_inv.dot(D)).t()

    def iterate_over_channs(self, input, output, n_channels):
        for i in range(n_channels):
            if self.method == NO_A:
                output[:, i, :] = input[:, i, :].mm(self.D_T)
            elif self.method in [BIN, WEI]:
                output[:, i, :] = input[:, i, :].mm(self.D_T).mm(self.A)
            else:
                raise RuntimeError('Unknown sampling method')
        return output

    def iterate_over_samples(self, input, output, n_samples):
        for i in range(n_samples):
            if self.method == NO_A:
                output[i, :, :] = input[i, :, :].mm(self.D_T)
            elif self.method in [BIN, WEI]:
                output[i, :, :] = input[i, :, :].mm(self.D_T).mm(self.A)
            else:
                raise RuntimeError('Unknown sampling method')
        return output

    # NOTE: forward an iterate functions has the exact same code! 
    def forward(self, input):
        if self.method == REG:
            sf = self.child_size/self.parent_size
            return nn.functional.interpolate(input, scale_factor=sf,
                                             mode='linear',
                                             align_corners=True)
        n_samples = input.shape[0]
        n_channs = input.shape[1]
        output = torch.zeros([n_samples, n_channs, self.child_size])
        if n_channs < n_samples:
            return self.iterate_over_channs(input, output, n_channs)
        else:
            return self.iterate_over_samples(input, output, n_samples)
