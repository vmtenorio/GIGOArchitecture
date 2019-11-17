import torch.nn as nn
import torch
from GIGO import layers

import numpy as np

DEBUG = False


class GIGOArch(nn.Module):
    def __init__(self,
                Si,         # GSO of the input graph
                So,         # GSO of the output graph
                Fi,         # Features in each graph filter layer of the input graph (list)
                Fo,         # Features in each graph filter layer of the output graph (list)
                Ki,         # Filter taps in each graph filter layer for the input graph
                Ko,         # Filter taps in each graph filter layer for the output graph
                C,          # Convolutional layers
                # M,          # Neurons in each fully connected layer (list)
                nonlin,     # Non linearity function
                batch_norm, # Whether or not to apply batch normalization
                arch_info):
        super(GIGOArch, self).__init__()
        # In python 3
        #super()

        # Define parameters
        if type(Si) != torch.FloatTensor:
            self.Si = torch.FloatTensor(Si)
        else:
            self.Si = Si
        if type(So) != torch.FloatTensor:
            self.So = torch.FloatTensor(So)
        else:
            self.So = So
        self.Ni = Si.shape[0]
        self.No = So.shape[0]
        self.Fi = Fi
        self.Fo = Fo
        self.Ki = Ki
        self.Ko = Ko
        self.C = C
        # self.M = M
        self.nonlin = nonlin
        self.batch_norm = batch_norm
        self.l_param = []

        # Some checks to verify data integrity
        assert self.Fi[-1] == self.No
        assert self.Fo[0] == self.Ni
        assert self.Fo[-1] == self.C[0]

        # Define the layers
        # Grahp Filter Layers for the input graph
        gfli = []
        for l in range(len(self.Fi)-1):
            # print("Graph filter layer: " + str(l))
            # print(str(self.F[l]) + ' x ' + str(self.F[l+1]))
            gfli.append(layers.GraphFilterUp(self.Si, self.Fi[l], self.Fi[l+1], self.Ki))
            gfli.append(self.nonlin())
            if self.batch_norm:
                gfli.append(nn.BatchNorm1d(self.Ni))
            self.l_param.append('weights_gfi_' + str(l))
            self.l_param.append('bias_gfi_' + str(l))

        self.GFLi = nn.Sequential(*gfli)

        # Grahp Filter Layers for the output graph
        gflo = []
        for l in range(len(self.Fo)-1):
            # print("Graph filter layer: " + str(l))
            # print(str(self.F[l]) + ' x ' + str(self.F[l+1]))
            gflo.append(layers.GraphFilterDown(self.So, self.Fo[l], self.Fo[l+1], self.Ko))
            # if l < len(self.Fo) -1:
            #     gflo.append(self.nonlin())
            gflo.append(self.nonlin())
            if self.batch_norm:
                gfli.append(nn.BatchNorm1d(self.No))
            self.l_param.append('weights_gfo_' + str(l))
            self.l_param.append('bias_gfo_' + str(l))

        self.GFLo = nn.Sequential(*gflo)

        self.conv1d = []
        for c in range(len(C) - 1):
            self.conv1d.append(nn.Conv1d(self.C[c], self.C[c+1], kernel_size=1, bias=True))
            self.conv1d.append(self.nonlin())
            self.l_param.append('weights_C_' + str(c))
            self.l_param.append('bias_c_' + str(c))

        self.conv1d_l = nn.Sequential(*self.conv1d)

        if arch_info:
            print("Architecture:")
            print("Input Graph N_nodes: {}, Output graph N_nodes: {}".format(self.Ni, self.No))
            print("Fin: {}, Fout: {}, Kin: {}, Kout: {}, C: {}".format(self.Fi, self.Fo, self.Ki, self.Ko, self.C))
            print("Non lin: " + str(self.nonlin))

    def forward(self, x):

        # Check type
        if type(x) != torch.FloatTensor:
            x = torch.FloatTensor(x)

        # Params
        T = x.shape[0]

        try:
            Fin = x.shape[1]
            xN = x.shape[2]
            assert Fin == self.Fi[0]
        except IndexError:
            xN = x.shape[1]
            Fin = 1
            x = x.unsqueeze(1)
            assert self.Fi[0] == 1

        assert xN == self.Ni

        # print('Starting')
        # print(x)
        # Define the forward pass
        # Graph filter layers
        # Goes from TxNxF[0] to TxNxF[-1] with GFL
        y = self.GFLi(x)
        # y shape should be T x Ni x No
        assert y.shape[2] == self.No

        y = y.permute(0, 2, 1)

        y = self.GFLo(y)
        # print('End')
        # print(y)
        y = self.conv1d_l(y)

        return y


class BasicArch(nn.Module):
    def __init__(self,
                S,
                F,          # Features in each graph filter layer (list)
                K,          # Filter taps in each graph filter layer
                M,          # Neurons in each fully connected layer (list)
                nonlin,     # Non linearity function
                arch_info): # Print architecture information
        super(BasicArch, self).__init__()
        # In python 3
        # super()

        # Define parameters
        if type(S) != torch.FloatTensor:
            self.S = torch.FloatTensor(S)
        else:
            self.S = S
        self.N = S.shape[0]
        self.F = F
        self.K = K
        self.M = M
        self.nonlin = nonlin
        self.l_param = []

        self.n_params = 0

        # Define the layer
        # Grahp Filter Layers
        gfl = []
        for l in range(len(self.F)-1):
            # print("Graph filter layer: " + str(l))
            # print(str(self.F[l]) + ' x ' + str(self.F[l+1]))
            gfl.append(layers.GraphFilter(self.S, self.F[l], self.F[l+1], self.K))
            gfl.append(self.nonlin())
            self.l_param.append('weights_gf_' + str(l))
            self.l_param.append('bias_gf_' + str(l))
            self.n_params += self.F[l] * self.F[l+1] * self.K + self.N * self.F[l+1]

        self.GFL = nn.Sequential(*gfl)

        # Fully connected Layers
        fcl = []
        # As last layer has no nonlin (if its softmax is done later, etc.)
        # define here the first layer before loop
        firstLayerIn = self.N*self.F[-1]
        fcl.append(nn.Linear(firstLayerIn, self.M[0]))
        self.l_param.append('weights_fc_0')
        self.l_param.append('bias_fc_0')
        for m in range(1,len(self.M)):
            # print("FC layer: " + str(m))
            # print(str(self.M[m-1]) + ' x ' + str(self.M[m]))
            fcl.append(self.nonlin())
            fcl.append(nn.Linear(self.M[m-1], self.M[m]))
            self.l_param.append('weights_fc_' + str(m))
            self.l_param.append('bias_fc_' + str(m))
            self.n_params += self.M[m-1] * self.M[m] + self.M[m]

        self.FCL = nn.Sequential(*fcl)

        if arch_info:
            print("Architecture:")
            print("Graph N_nodes: {}".format(self.N))
            print("F: {}, K: {}, M: {}".format(self.F, self.K, self.M))
            print("Non lin: " + str(self.nonlin))
            print("N params: " + str(self.n_params))

    def forward(self, x):

        #Check type
        if type(x) != torch.FloatTensor:
            x = torch.FloatTensor(x)

        # Params
        T = x.shape[0]
        xN = x.shape[1]

        assert xN == self.N

        try:
            Fin = x.shape[2]
            assert Fin == self.F[0]
        except IndexError:
            Fin = 1
            x = x.unsqueeze(2)
            assert self.F[0] == 1

        # Define the forward pass
        # Graph filter layers
        # Goes from TxNxF[0] to TxNxF[-1] with GFL
        y = self.GFL(x)

        # return y.squeeze(2)

        y = y.reshape([T, self.N*self.F[-1]])

        return self.FCL(y)


class MLP(nn.Module):
    def __init__(self, F, nonlin, arch_info):
        super(MLP, self).__init__()
        self.F = F
        self.nonlin = nonlin

        layers = []
        self.l_param = []
        self.n_params = 0
        for l in range(len(self.F)-1):
            layers.append(nn.Linear(self.F[l], self.F[l+1]))
            if self.nonlin != None:
                layers.append(self.nonlin())
            self.l_param.append('weights_' + str(l))
            self.l_param.append('bias_' + str(l))
            self.n_params += self.F[l] * self.F[l+1] + self.F[l+1]
        self.MLP = nn.Sequential(*layers)

        if arch_info:
            print("Multi-Layer Perceptron architecture")
            print("Features: " + str(self.F))
            print("Non lin: " + str(self.nonlin))
            print("N params: " + str(self.n_params))

    def forward(self, x):
        #Check type
        if type(x) != torch.FloatTensor:
            x = torch.FloatTensor(x)
        return self.MLP(x)

class ConvNN(nn.Module):
    def __init__(self, N, F, kernel_size, nonlin, arch_info):
        super(ConvNN, self).__init__()
        self.N = N  # For length calculation
        self.F = F
        self.kernel_size = kernel_size
        self.nonlin = nonlin

        layers = []
        self.l_param = []
        self.n_params = 0
        for l in range(len(self.F)-1):
            layers.append(nn.Conv1d(self.F[l], self.F[l+1], self.kernel_size))
            layers.append(self.nonlin())
            self.l_param.append('weights_' + str(l))
            self.l_param.append('bias_' + str(l))
            self.n_params += self.F[l] * self.F[l+1] * self.kernel_size + self.F[l+1]
        self.conv1d_nl = nn.Sequential(*layers)

        self.delta_len = (self.kernel_size - 1) * (len(self.F)-1)
        self.MLP = nn.Linear((self.N - self.delta_len)*self.F[-1], self.N)

        if arch_info:
            print("Convolutional architecture")
            print("Features: {}, Kernel size: {}".format(str(self.F), self.kernel_size))
            print("Non lin: " + str(self.nonlin))
            print("N params: " + str(self.n_params))

    def forward(self, x):
        #Check type
        if type(x) != torch.FloatTensor:
            x = torch.FloatTensor(x)

        T = x.shape[0]
        try:
            N = x.shape[2]
            F = x.shape[1]
            assert F == self.F[0]
        except IndexError:
            F = 1
            x = x.unsqueeze(1)
            assert self.F[0] == 1

        y = self.conv1d_nl(x)
        T, C, L = y.shape

        assert C == self.F[-1]
        assert L == self.N - self.delta_len

        y = y.reshape([T, L*C])

        return self.MLP(y)
