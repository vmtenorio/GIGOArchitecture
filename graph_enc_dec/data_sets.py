import numpy as np
import matplotlib.pyplot as plt
from pygsp.graphs import Graph, StochasticBlockModel, ErdosRenyi
from torch import Tensor

from math import pi
# Graph Type Constants
SBM = 1
ER = 2

# Comm Node Assignment Constants
CONT = 1    # Contiguous nodes
ALT = 2    # Alternated nodes
RAND = 3    # Random nodes


def assign_nodes_to_comms(N,k):
    """
    Distribute contiguous nodes in the same community while assuring that all
    communities have (approximately) the same number of nodes.
    """
    z = np.zeros(N, dtype=np.int)
    leftover_nodes = N % k
    grouped_nodes = 0
    for i in range(k):
        if leftover_nodes > 0:
            n_nodes = np.ceil(N/k).astype(np.int)
            leftover_nodes -= 1
        else:
            n_nodes = np.floor(N/k).astype(np.int)
        z[grouped_nodes:(grouped_nodes+n_nodes)] = i
        grouped_nodes += n_nodes
    return z


def create_graph(ps, seed=None):
    """
    Create a random graph using the parameters specified in the dictionary ps.
    The keys that this dictionary should nclude are:
        - type: model for the graph. Options are SBM (Stochastic Block Model)
          or ER (Erdos-Renyi)
        - N: number of nodes
        - k: number of communities (for SBM only)
        - p: edge probability for nodes in the same community
        - q: edge probability for nodes in different communities (for SBM only)
        - type_z: specify how to assigns nodes to communities (for SBM only).
          Options are CONT (continous), ALT (alternating) and RAND (random)
    """
    if ps['type'] == SBM:
        if ps['type_z'] == CONT:
            z = assign_nodes_to_comms(ps['N'],ps['k'])
        elif ps['type_z'] == ALT:
            z = np.array(list(range(ps['k']))*int(ps['N']/ps['k'])+list(range(ps['N']%ps['k'])))
        elif ps['type_z'] == RAND:
            z = assign_nodes_to_comms(ps['N'],ps['k'])
            np.random.shuffle(z)
        else:
            z = None
        return StochasticBlockModel(N=ps['N'], k=ps['k'], p=ps['p'], z=z,
                                    q=ps['q'], connected=True, seed=seed)
    elif ps['type'] == ER:
        return ErdosRenyi(N=ps['N'], p=ps['p'], connected=True, seed=seed)
    else:
        raise RuntimeError('Unknown graph type')


def perturbated_graphs(g_params, eps_c, eps_d, seed=None):
    """
    Create 2 closely related graphs. The first graph is created following the
    indicated model and the second is a perturbated version of the previous
    where links are created or destroid with a small probability.
    Arguments:
        - g_params: a dictionary containing all the parameters for creating
          the desired graph. The options are explained in the documentation
          of the function 'create_graph'
        - eps_c: probability for creating new edges
        - eps_d: probability for destroying existing edges
    """
    Gx = create_graph(g_params, seed)
    A_x = Gx.A.todense()
    # TODO: make a test for ensuring both graphs have same nodes, comms, etc
    # they should only differ on the links and both should be sym
    link_ind = np.where(A_x == 1)
    no_link_ind = np.where(A_x != 1)
    mask_c = np.random.choice([0, 1], p=[1-eps_c, eps_c], size=no_link_ind[0].shape)
    mask_d = np.random.choice([0, 1], p=[1-eps_d, eps_d], size=link_ind[0].shape)

    A_y = A_x
    A_y[link_ind] = np.logical_xor(A_y[link_ind], mask_d).astype(int)
    A_y[no_link_ind] = np.logical_xor(A_y[no_link_ind], mask_c).astype(int)
    A_y = np.triu(A_y, 1)
    A_y = A_y + A_y.T

    Gy = Graph(A_y)
    if not Gy.is_connected():
        raise RuntimeError('Could not create connected graph Gy')
    Gx.set_coordinates('community2D')
    Gy.set_coordinates(Gx.coords)
    Gy.info = {'node_com': Gx.info['node_com'], 'comm_sizes': Gx.info['comm_sizes']}
    return Gx, Gy


class DiffusedSparse2GS:
    """
    Class for generating graph signals X and Y following two diffusion processes
    generated over two different graphs. However, the sparse signal which will
    be diffused is the same in both cases.
    The two graphs must have the same number of nodes and communities.
    Arguments:
        - Gx: graph where signal X will be defined
        - Gy: graph where signal Y will be defined
        - n_samples: a list with the number of samples for training, validation
          and test. Alternatively, if only an integer is provided
    """
    def __init__(self, Gx, Gy, n_samples, L, n_delts, min_d=-1, max_d=1):
        if Gx.N != Gy.N:
            raise RuntimeError('Both graphs must have the same number of nodes')
        if Gx.info['comm_sizes'].size != Gy.info['comm_sizes'].size:
            raise RuntimeError('Both graphs must have the same number of communities')

        if not isinstance(n_samples, list):
            self.n_train = n_samples
            self.n_val = int(np.floor(0.2*n_samples))
            self.n_test = int(np.floor(0.2*n_samples))
        elif len(n_samples) == 3:
            self.n_train = n_samples[0]
            self.n_val = n_samples[1]
            self.n_test = n_samples[2]
        else:
            raise RuntimeError('n_samples must be an integer or a list with the \
                                samples for training, validation and test')
        self.Gx = Gx
        self.Gy = Gy

    def median_neighbours_nodes(self, X, G):
        X_aux = np.zeros(X.shape)
        X = np.tanh(X)
        for i in range(G.N):
            _, neighbours = np.asarray(G.W.todense()[i, :] != 0).nonzero()
            X_aux[:, i] = np.median(X[:, neighbours], axis=1)
        return X_aux

    def to_tensor(self, n_chans=1):
        N = self.train_X.shape[1]
        self.train_X = Tensor(self.train_X).view([self.n_train, n_chans, N])
        self.train_Y = Tensor(self.train_Y).view([self.n_train, n_chans, N])
        self.val_X = Tensor(self.val_X).view([self.n_val, n_chans, N])
        self.val_Y = Tensor(self.val_Y).view([self.n_val, n_chans, N])
        self.test_X = Tensor(self.test_X).view([self.n_test, n_chans, N])
        self.test_Y = Tensor(self.test_Y).view([self.n_test, n_chans, N])

    def to_unit_norm(self):
        self.train_X = self._to_unit_norm(self.train_X)
        self.train_Y = self._to_unit_norm(self.train_Y)
        self.val_X = self._to_unit_norm(self.val_X)
        self.val_Y = self._to_unit_norm(self.val_Y)
        self.test_X = self._to_unit_norm(self.test_X)
        self.test_Y = self._to_unit_norm(self.test_Y)

    def _to_unit_norm(self, signals):
        """
        Divide each signal by its norm so all samples have unit norm
        """
        norm = np.sqrt(np.sum(signals**2,axis=1))
        if 0 in norm:
            print("WARNING: signal with norm 0")
            return None
        return (signals.T/norm).T

    def sparse_S(self, n_samp, n_deltas, min_delta, max_delta):
        """
        Create random sparse signal s composed of different deltas placed in the
        different communities of the graph. If the graph is an ER, then deltas
        are just placed on random nodes
        """
        S = np.zeros((self.Gx.N, n_samp))

        # Create delta mean values
        n_comms = self.Gx.info['comm_sizes'].size
        if n_comms > 1:
            step = (max_delta-min_delta)/(n_comms-1)
        else:
            step = (max_delta-min_delta)/(n_deltas-1)
        ds_per_comm = np.ceil(n_deltas/n_comms).astype(int)
        delta_means = np.arange(min_delta, max_delta+0.1, step)
        delta_means = np.tile(delta_means, ds_per_comm)[:n_deltas]        
        for i in range(n_samp):
            delta_values = np.random.randn(n_deltas)*step/4 + delta_means
            # Randomly assign delta value to comm nodes
            for j in range(n_deltas):
                delta = delta_values[j]
                com_j = j % self.Gx.info['comm_sizes'].size
                com_nodes, = np.asarray(self.Gx.info['node_com'] == com_j).nonzero()
                rand_index = np.random.randint(0, self.Gx.info['comm_sizes'][com_j])
                S[com_nodes[rand_index], i] = delta
        return S.T


class LinearDS2GS(DiffusedSparse2GS):
    def __init__(self, Gx, Gy, n_samples, L, n_delts, min_d=-1,
                 max_d=1, median=False):
        super(LinearDS2GS, self).__init__(Gx, Gy, n_samples, L, n_delts, min_d,
                                          max_d)
        self.random_diffusing_filters(L)

        # Create samples
        self.train_S = self.sparse_S(self.n_train, n_delts, min_d, max_d)
        self.train_X = self.Hx.dot(self.train_S.T).T
        self.train_Y = self.Hy.dot(self.train_S.T).T
        self.val_S = self.sparse_S(self.n_val, n_delts, min_d, max_d)
        self.val_X = self.Hx.dot(self.val_S.T).T
        self.val_Y = self.Hy.dot(self.val_S.T).T
        self.test_S = self.sparse_S(self.n_test, n_delts, min_d, max_d)
        self.test_X = self.Hx.dot(self.test_S.T).T
        self.test_Y = self.Hy.dot(self.test_S.T).T

        if median:
            self.train_X = self.median_neighbours_nodes(self.train_X, self.Gx)
            self.train_Y = self.median_neighbours_nodes(self.train_Y, self.Gy)
            self.val_X = self.median_neighbours_nodes(self.val_X, self.Gx)
            self.val_Y = self.median_neighbours_nodes(self.val_Y, self.Gy)
            self.test_X = self.median_neighbours_nodes(self.test_X, self.Gx)
            self.test_Y = self.median_neighbours_nodes(self.test_Y, self.Gy)

    def random_diffusing_filters(self, L):
        """
        Create two lineal random diffusing filters with L random coefficients
        using the graphs shift operators from Gx and Gy.
        Arguments:
            - L: number of filter coeffcients
        """
        hs_x = np.random.rand(L)
        hs_y = np.random.rand(L)
        self.Hx = np.zeros(self.Gx.W.shape)
        self.Hy = np.zeros(self.Gy.W.shape)
        Sx = self.Gx.W.todense()
        Sy = self.Gy.W.todense()
        for l in range(L):
            self.Hx += hs_x[l]*np.linalg.matrix_power(Sx, l)
            self.Hy += hs_y[l]*np.linalg.matrix_power(Sy, l)


class NonLinearDS2GS(DiffusedSparse2GS):
    def __init__(self, Gx, Gy, n_samples, L, n_delts, min_d=-1,
                 max_d=1, median=False):
        super(NonLinearDS2GS, self).__init__(Gx, Gy, n_samples, L, n_delts,
                                             min_d, max_d)
        self.train_S = self.sparse_S(self.n_train, n_delts, min_d, max_d)
        self.val_S = self.sparse_S(self.n_val, n_delts, min_d, max_d)
        self.test_S = self.sparse_S(self.n_test, n_delts, min_d, max_d)
        self.train_X = self.signal_diffusion(self.train_S, self.Gx, L)
        self.train_Y = self.signal_diffusion(self.train_S, self.Gy, L)
        self.val_X = self.signal_diffusion(self.val_S, self.Gx, L)
        self.val_Y = self.signal_diffusion(self.val_S, self.Gy, L)
        self.test_X = self.signal_diffusion(self.test_S, self.Gx, L)
        self.test_Y = self.signal_diffusion(self.test_S, self.Gy, L)

        if median:
            self.train_X = self.median_neighbours_nodes(self.train_X, self.Gx)
            self.train_Y = self.median_neighbours_nodes(self.train_Y, self.Gy)
            self.val_X = self.median_neighbours_nodes(self.val_X, self.Gx)
            self.val_Y = self.median_neighbours_nodes(self.val_Y, self.Gy)
            self.test_X = self.median_neighbours_nodes(self.test_X, self.Gx)
            self.test_Y = self.median_neighbours_nodes(self.test_Y, self.Gy)

    def signal_diffusion(self, S, G, L):
        S_T = S.T
        X_T = np.zeros(S_T.shape)
        A = G.W.todense()
        h = np.random.rand(L)
        for l in range(L):
            X_T += h[l]*np.linalg.matrix_power(A, l).dot(np.sign(S_T)*S_T**l)
        return X_T.T
