import torch.nn as nn
import torch
from GIGO import layers

import numpy as np

class BasicArch(nn.Module):
    def __init__(self,
                S,
                F,          # Features in each graph filter layer (list)
                K,          # Filter taps in each graph filter layer
                M,          # Neurons in each fully connected layer (list)
                nonlin):    # Non linearity function
        super(BasicArch, self).__init__()
        # In python 3
        #super()

        # Define parameters
        if type(S) == np.ndarray:
            self.S = torch.FloatTensor(S)
        else:
            self.S = S
        self.N = S.shape[0]
        self.F = F
        self.K = K
        self.M = M
        self.nonlin = nonlin
        self.l_param = []

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

        self.FCL = nn.Sequential(*fcl)


    def forward(self, x):

        #Check type
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).type(torch.FloatTensor)

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

        #return y.squeeze(2)

        y = y.reshape([T, self.N*self.F[-1]])

        return self.FCL(y)

class GIGOArch(nn.Module):
    def __init__(self,
                Si,         # GSO of the input graph
                So,         # GSO of the output graph
                Fi,         # Features in each graph filter layer of the input graph (list)
                Fo,         # Features in each graph filter layer of the output graph (list)
                Ki,         # Filter taps in each graph filter layer for the input graph
                Ko,         # Filter taps in each graph filter layer for the output graph
                #M,          # Neurons in each fully connected layer (list)
                nonlin):    # Non linearity function)
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
        #self.M = M
        self.nonlin = nonlin
        self.l_param = []

        # Some checks to verify data integrity
        assert self.Fi[-1] == self.No
        assert self.Fo[0] == self.Ni

        # Define the layers
        # Grahp Filter Layers for the input graph
        gfli = []
        for l in range(len(self.Fi)-1):
            # print("Graph filter layer: " + str(l))
            # print(str(self.F[l]) + ' x ' + str(self.F[l+1]))
            gfli.append(layers.GraphFilter(self.Si, self.Fi[l], self.Fi[l+1], self.Ki))
            gfli.append(self.nonlin())
            self.l_param.append('weights_gfi_' + str(l))
            self.l_param.append('bias_gfi_' + str(l))

        self.GFLi = nn.Sequential(*gfli)

        # Grahp Filter Layers for the output graph
        gflo = []
        for l in range(len(self.Fo)-1):
            # print("Graph filter layer: " + str(l))
            # print(str(self.F[l]) + ' x ' + str(self.F[l+1]))
            gflo.append(layers.GraphFilter(self.So, self.Fo[l], self.Fo[l+1], self.Ko))
            gflo.append(self.nonlin())
            self.l_param.append('weights_gfo_' + str(l))
            self.l_param.append('bias_gfo_' + str(l))

        self.GFLo = nn.Sequential(*gflo)

    def forward(self, x):

        #Check type
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).type(torch.FloatTensor)

        # Params
        T = x.shape[0]
        xN = x.shape[1]

        assert xN == self.Ni

        try:
            Fin = x.shape[2]
            assert Fin == self.F[0]
        except IndexError:
            Fin = 1
            x = x.unsqueeze(2)
            assert self.Fi[0] == 1

        #print('Starting')
        #print(x)
        # Define the forward pass
        # Graph filter layers
        # Goes from TxNxF[0] to TxNxF[-1] with GFL
        y = self.GFLi(x)
        # y shape should be T x Ni x No
        assert y.shape[2] == self.No
        print('Intermediate')
        print(y)

        y = y.permute(0,2,1)
        print('Intermediate2')
        print(y)

        y = self.GFLo(y)
        #print('End')
        #print(y)

        return torch.squeeze(y, dim=2)
