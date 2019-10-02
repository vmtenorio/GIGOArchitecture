from torch import manual_seed, nn, Tensor, optim, no_grad
import copy
import torch
import sys
import numpy as np
import time

from graph_clustering import NONE, REG, NO_A, BIN, WEI


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
                 As_enc=None, As_dec=None, gamma=0.5, batch_norm=True,
                 # Activation functions
                 act_fn=nn.Tanh(), last_act_fn=nn.Tanh()):
        if features_enc[-1] != features_dec[0] or nodes_enc[-1] != nodes_dec[0]:
            raise RuntimeError('Different definition of dimension for the latent space')

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
                # TODO: Need to notify if reg_ups will be used!
                # TODO: Create reg_downs --> are this two really needed in this setting??
                # A = None if self.downs == 'no_A' or self.downs == 'original' else self.hier_A[l+1]
                self.add_layer(GraphDownsampling(self.Ds[l-downs_skip]))
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
                # TODO: Need to notify if reg_ups or no_A will be used! 
                # Add layer graph upsampling
                # ups?
                # Careful, A may be None
                # self.add_layer(GraphUpsampling(self.Us[l-ups_skip], A, self.ups, self.gamma))
                self.add_layer(GraphUpsampling(self.Us[l-ups_skip], self.As_dec[l+1-ups_skip],
                                               self.gamma))
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


class GraphUpsampling(nn.Module):
    """
    Use information from the agglomerative hierarchical clustering for doing the upsampling by
    creating the upsampling matrix U
    """
    def __init__(self, U, A, gamma, method=WEI):  #WEI
        super(GraphUpsampling, self).__init__()
        # NOTE: Normalize A so its rows add up to 1 --> maybe should be done when obtainning A
        if A is not None:
            D_inv = np.linalg.inv(np.diag(np.sum(A,0)))
            self.A = D_inv.dot(A)
        self.parent_size = U.shape[1]
        self.child_size = U.shape[0]
        self.method = method
        self.gamma = gamma
        self.U = Tensor(U)

    def forward(self, input):
        # TODO: check if making ops with np instead of torch increase speed
        n_samples = input.shape[0]
        n_channels = input.shape[1]
        output = torch.zeros([n_samples, n_channels, self.U.shape[0]])
        for i in range(n_samples):
            in_matrix = torch.t(input[i,:,:])
            parents_val = self.U.mm(in_matrix)
            # NOTE: gamma = 1 is equivalent to no_A
            # NOTE: gamma = 0 is equivalent to the prev setup
            if self.method == REG:
                sf = self.child_size/self.parent_size
                return nn.functional.interpolate(input, scale_factor=sf,
                                        mode='linear', align_corners=True)
            elif self.method == NO_A:
                output[i,:,:] = torch.t(parents_val)
            elif self.method in [BIN, WEI]:
                neigbours_val = Tensor(self.A).mm(parents_val)
                output[i,:,:] = torch.t(self.gamma*parents_val + (1-self.gamma)*neigbours_val)
            else:
                raise RuntimeError('Unknown sampling method')

        return output


# TODO: add function for printing its info correctly
class GraphDownsampling(nn.Module):
    def __init__(self, D):
        # Maybe different types of Ds?
        super(GraphDownsampling, self).__init__()
        # Normalize D so all its rows add to 1
        # Maybe this should depend on the different As
        Deg_inv = np.linalg.inv(np.diag(np.sum(D,1)))  
        self.D = Tensor(Deg_inv.dot(D))

    def forward(self, input):
        n_samples = input.shape[0]
        output = torch.zeros([n_samples, input.shape[1], self.D.shape[0]])
        for i in range(n_samples):
            in_matrix = torch.t(input[i,:,:])
            output[i,:,:] = torch.t(self.D.mm(in_matrix))
        return output
