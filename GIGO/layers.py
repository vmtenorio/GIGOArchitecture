import torch
import torch.nn as nn
import math

import numpy as np

DEBUG = False


class GraphFilter(nn.Module):
    def __init__(self,
                 # GSO
                 S,
                 # Size of the filter
                 Fin, Fout, K
                 ):
        super(GraphFilter, self).__init__()
        self.S = S
        self.N = S.shape[0]
        self.Fin = Fin
        self.Fout = Fout
        self.K = K

        self.Spow = torch.zeros(K, self.N, self.N)
        self.Spow[0, :, :] = torch.eye(self.N)
        for i in range(1, K):
            self.Spow[i, :, :] = torch.matmul(self.Spow[i-1, :, :], self.S)

    def calc_filter(self, fout):
        return (self.weights[:, fout].view([self.K, 1, 1])*self.Spow).sum(0)


class GraphFilterUp(GraphFilter):
    def __init__(self,
                 # GSO
                 S,
                 # Size of the filter
                 Fin, Fout, K
                 ):
        super(GraphFilterUp, self).__init__(S, Fin, Fout, K)

        assert Fout % Fin == 0
        self.mult = Fout // Fin

        # self.weights = nn.Parameter(torch.Tensor(self.K, self.Fout))
        self.weights = nn.Parameter(torch.Tensor(self.K, self.Fout))

        stdv = 1. / math.sqrt(self.Fin * self.K)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x shape T x Fin x N
        T = x.shape[0]
        xFin = x.shape[1]
        xN = x.shape[2]

        assert xN == self.N
        assert xFin == self.Fin

        y = torch.zeros(T, self.Fout, self.N)
        fIn = 0
        for f in range(self.Fout):
            xF = x[:, fIn, :]
            # No need to transpose H because its symmetric
            H = self.calc_filter(f)
            y[:, f, :] = torch.matmul(xF, H)

            if f % self.mult == self.mult - 1:
                fIn += 1
        return y


class GraphFilterDown(GraphFilter):
    def __init__(self,
                 # GSO
                 S,
                 # Size of the filter
                 Fin, Fout, K):
        super(GraphFilterDown, self).__init__(S, Fin, Fout, K)

        assert Fin % Fout == 0
        self.mult = Fin // Fout

        self.weights = nn.Parameter(torch.Tensor(self.K, self.Fin))
        stdv = 1. / math.sqrt(self.Fin * self.K)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x shape T x Fin x N
        T = x.shape[0]
        xFin = x.shape[1]
        xN = x.shape[2]

        assert xN == self.N
        assert xFin == self.Fin

        y = torch.zeros(T, self.Fout, self.N)
        for f in range(self.Fout):
            # y_aux = torch.zeros(T, self.N)
            for m in range(self.mult):
                fIn = self.mult * f + m
                xF = x[:, fIn, :]
                H = self.calc_filter(fIn)
                y_aux += torch.matmul(xF, H)/2 
            y[:, f, :] = y_aux

        return y


class GraphFilterSelective(nn.Module):

    def __init__(self,
                 # GSO
                 S,
                 # Size of the filter
                 Fin, Fout, K):
        raise NotImplementedError
        super(GraphFilterSelective, self).__init__(self)
        self.S = S
        self.N = S.shape[0]
        self.Fin = Fin
        self.Fout = Fout
        self.K = K

        if Fin > Fout:
            assert Fin % Fout == 0
            self.mult = Fin // Fout
        else:
            assert Fout % Fin == 0
            self.mult = Fout // Fin

        self.weights = nn.parameter.Parameter(torch.Tensor(self.K, self.Fout))
        stdv = 1. / math.sqrt(self.Fin * self.K)
        self.weights.data.uniform_(-stdv, stdv)

        # Calculating powers of GSO
        Spow = [torch.eye(self.N)]
        for i in range(1, K):
            Spow.append(torch.matmul(Spow[-1], self.S))

    def forward(self, x):
        # x shape T x Fin x N
        T = x.shape[0]
        xFin = x.shape[1]
        xN = x.shape[2]

        assert xN == self.N
        assert xFin == self.Fin

        y = torch.zeros(T, self.Fout, self.N)

        return y


class GraphFilterFC(nn.Module):

    def __init__(self,
                # GSO
                S,
                # Size of the filter
                Fin, Fout, K
                ):
        super(GraphFilterFC, self).__init__()
        self.S = S      # Graph Filter
        self.N = S.shape[0]
        self.Fin = Fin
        self.Fout = Fout
        self.K = K

        # Fernando's weight Initialization
        self.weights = nn.parameter.Parameter(torch.Tensor(self.Fin*self.K, self.Fout))
        stdv = 1. / math.sqrt(self.Fin * self.K)
        self.weights.data.uniform_(-stdv, stdv)
        # self.bias = nn.parameter.Parameter(torch.Tensor(1, self.N, self.Fout))
        # self.bias.data.uniform_(-stdv, stdv)
        torch.set_printoptions(threshold=100)

    def forward(self, x):
        # x shape T x N x Fin
        # Graph filter
        T = x.shape[0]
        xN = x.shape[1]
        xFin = x.shape[2]
        if DEBUG:
            print('Fin: ' + str(xFin))
            print('X beg-')
            print(x)
            print('Weights - ')
            print(self.weights)
            print('Bias - ')
            # print(self.bias)
            print('Graph - ')
            print(self.S)

        assert xN == self.N
        assert xFin == self.Fin

        x = x.permute(1, 2, 0)      # N x Fin x T
        x = x.reshape([self.N, self.Fin*T])
        x_list = []
        Spow = torch.eye(self.N)

        for k in range(self.K):
            x1 = torch.matmul(Spow, x)
            x_list.append(x1)
            if DEBUG:
                print(x1)
                print(x1.shape)
            Spow = torch.matmul(Spow, self.S)
            if DEBUG and False:
                print('Power ' + str(k))
                print(Spow)
        # x shape after loop: K x N x Fin*T
        x = torch.stack(x_list)
        if DEBUG:
            print('X - 1')
            print(x)

        x = x.reshape([self.K, self.N, self.Fin, T])
        x = x.permute(3,1,2,0)
        x = x.reshape([T*self.N, self.K*self.Fin])
        if DEBUG:
            print('X - 2')
            print(x)
            print('Weights - ')
            print(self.weights)

        # Apply weights
        y = torch.matmul(x, self.weights)       # y shape: T*N x Fout
        if DEBUG:
            print('Y - 1')
            print(y)

        y = y.reshape([T, self.N, self.Fout])
        if DEBUG:
            print('Y before bias-')
            print(y.shape)
            print(y)
        # y = y + self.bias
        if DEBUG:
            print('Y end-')
            print(y.shape)
            print(y)
        return y
