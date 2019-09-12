import torch
import torch.nn as nn
import math

class GraphFilter(nn.Module):

    def __init__(self,
                # GSO
                S,
                # Size of the filter
                Fin, Fout, K
                ):
        super(GraphFilter, self).__init__()
        self.S = S      # Graph Filter
        self.N = S.shape[0]
        self.Fin = Fin
        self.Fout = Fout
        self.K = K

        # Fernando's weight Initialization
        self.weights = nn.parameter.Parameter(torch.Tensor(self.Fin*self.K, self.Fout))
        stdv = 1. / math.sqrt(self.Fin * self.K)
        self.weights.data.uniform_(-stdv, stdv)
        self.bias = nn.parameter.Parameter(torch.Tensor(1, self.N, self.Fout))
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x shape T x N x Fin
        # Graph filter
        T = x.shape[0]
        xN = x.shape[1]
        xFin = x.shape[2]
        # print('X beg-')
        # print(x)

        assert xN == self.N
        assert xFin == self.Fin

        x = x.permute(1, 2, 0)      # N x Fin x T
        x = x.reshape([self.N, self.Fin*T])
        x_list = []
        Spow = torch.ones([self.N, self.N])

        for k in range(self.K):
            x1 = torch.matmul(Spow, x)
            x_list.append(x1)
            Spow = torch.matmul(Spow, self.S)
        # x shape after loop: K x N x Fin*T
        x = torch.stack(x_list)

        x = x.reshape([self.K, self.N, self.Fin, T])
        x = x.permute(3,1,2,0)
        x = x.reshape([T*self.N, self.K*self.Fin])

        # Apply weights
        y = torch.matmul(x, self.weights)       # y shape: T*N x Fout

        y = y.reshape([T, self.N, self.Fout])
        y = y + self.bias
        # print('Y end-')
        # print(y.shape)
        # print(y)
        return y