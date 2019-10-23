import numpy as np
import matplotlib.pyplot as plt
from pygsp.graphs import Graph, StochasticBlockModel, ErdosRenyi
from torch import Tensor

# Graph Type Constants
SBM = 1
ER = 2

# Comm Node Assignment Constants
CONT = 1    # Contiguous nodes
ALT = 2    # Alternated nodes
RAND = 3    # Random nodes


def assign_nodes_to_comms(N, k):
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
            z = assign_nodes_to_comms(ps['N'], ps['k'])
        elif ps['type_z'] == ALT:
            z = np.array(list(range(ps['k']))*int(ps['N']/ps['k']) +
                         list(range(ps['N'] % ps['k'])))
        elif ps['type_z'] == RAND:
            z = assign_nodes_to_comms(ps['N'], ps['k'])
            np.random.shuffle(z)
        else:
            z = None
        return StochasticBlockModel(N=ps['N'], k=ps['k'], p=ps['p'], z=z,
                                    q=ps['q'], connected=True, seed=seed,
                                    max_iter=25)
    elif ps['type'] == ER:
        return ErdosRenyi(N=ps['N'], p=ps['p'], connected=True, seed=seed,
                          max_iter=25)
    else:
        raise RuntimeError('Unknown graph type')


def perturbate_probability(Gx, eps_c, eps_d):
    A_x = Gx.W.todense()
    no_link_ind = np.where(A_x == 0)
    link_ind = np.where(A_x == 1)

    mask_c = np.random.choice([0, 1], p=[1-eps_c, eps_c],
                              size=no_link_ind[0].shape)
    mask_d = np.random.choice([0, 1], p=[1-eps_d, eps_d],
                              size=link_ind[0].shape)

    A_x[link_ind] = np.logical_xor(A_x[link_ind], mask_d).astype(int)
    A_x[no_link_ind] = np.logical_xor(A_x[no_link_ind], mask_c).astype(int)
    A_x = np.triu(A_x, 1)
    A_y = A_x + A_x.T
    return A_y


def perturbate_percentage(Gx, creat, destr):
    A_x_triu = Gx.W.todense()
    A_x_triu[np.tril_indices(Gx.N)] = -1

    # Create links
    no_link_i = np.where(A_x_triu == 0)
    links_c = np.random.choice(no_link_i[0].size, int(Gx.Ne * creat/100))
    idx_c = (no_link_i[0][links_c], no_link_i[1][links_c])

    # Destroy links
    link_i = np.where(A_x_triu == 1)
    links_d = np.random.choice(link_i[0].size, int(Gx.Ne * destr/100))
    idx_d = (link_i[0][links_d], link_i[1][links_d])

    A_x_triu[np.tril_indices(Gx.N)] = 0
    A_x_triu[idx_c] = 1
    A_x_triu[idx_d] = 0
    A_y = A_x_triu + A_x_triu.T
    return A_y


def perturbated_graphs(g_params, eps_c=5, eps_d=5, pct=True, seed=None):
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
    if pct:
        A_y = perturbate_percentage(Gx, eps_c, eps_d)
    else:
        A_y = perturbate_probability(Gx, eps_c, eps_d)

    Gy = Graph(A_y)
    assert Gy.is_connected(), 'Could not create connected graph Gy'
    Gx.set_coordinates('community2D')
    Gy.set_coordinates(Gx.coords)
    Gy.info = {'node_com': Gx.info['node_com'],
               'comm_sizes': Gx.info['comm_sizes']}
    return Gx, Gy


class DiffusedSparse2GS:
    """
    Class for generating graph signals X and Y following two diffusion
    processes generated over two different graphs. However, the sparse
    signal which will be diffused is the same in both cases. The two
    graphs must have the same number of nodes and communities.
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
        for i in range(G.N):
            _, neighbours = np.asarray(G.W.todense()[i, :] != 0).nonzero()
            X_aux[:, i] = np.median(X[:, np.append(neighbours, i)], axis=1)
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
        norm = np.sqrt(np.sum(signals**2, axis=1))
        if 0 in norm:
            print("WARNING: signal with norm 0")
            return None
        return (signals.T/norm).T

    def plot_train_signals(self, ids, show=True):
        if not isinstance(ids, list) and not isinstance(ids, range):
            ids = [ids]
        for id in ids:
            S = self.train_S[id, :]
            X = self.train_X[id, :]
            Y = self.train_Y[id, :]
            _, axes = plt.subplots(2, 2)
            self.Gx.plot_signal(S, ax=axes[0, 0])
            self.Gx.plot_signal(X, ax=axes[0, 1])
            self.Gy.plot_signal(S, ax=axes[1, 0])
            self.Gy.plot_signal(Y, ax=axes[1, 1])
        if show:
            plt.show()

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

    def add_noise(self, p_n, test_only=True):
        if p_n == 0:
            return
        self.test_X = self.add_noise_to_X(self.test_X, p_n)
        self.test_Y = self.add_noise_to_X(self.test_Y, p_n)

        if test_only:
            return
        self.val_X = self.add_noise_to_X(self.val_X, p_n)
        self.val_Y = self.add_noise_to_X(self.val_Y, p_n)
        self.train_X = self.add_noise_to_X(self.train_X, p_n)
        self.train_Y = self.add_noise_to_X(self.train_Y, p_n)

    def add_noise_to_X(self, X, p_n):
        p_x = np.sum(X**2, axis=1)
        sigma = np.sqrt(p_n * p_x / X.shape[1])
        Noise = (np.random.randn(X.shape[0], X.shape[1]).T * sigma).T
        return X + Noise


class LinearDS2GS(DiffusedSparse2GS):
    def __init__(self, Gx, Gy, n_samples, L, n_delts, min_d=-1,
                 max_d=1, median=True, same_coeffs=False):
        super(LinearDS2GS, self).__init__(Gx, Gy, n_samples, L, n_delts, min_d,
                                          max_d)
        self.median = median
        self.hx = np.random.rand(L)
        self.hy = self.hx if same_coeffs else np.random.rand(L)
        self.random_diffusing_filters()
        self.train_S = self.sparse_S(self.n_train, n_delts, min_d, max_d)
        self.val_S = self.sparse_S(self.n_val, n_delts, min_d, max_d)
        self.test_S = self.sparse_S(self.n_test, n_delts, min_d, max_d)
        self.create_samples_X_Y()

    def state_dict(self):
        state = {}
        state['train_S'] = self.train_S
        state['val_S'] = self.val_S
        state['test_S'] = self.test_S
        state['hx'] = self.hx
        state['hy'] = self.hy
        state['median'] = self.median
        return state

    def load_state_dict(self, state):
        self.train_S = state['train_S']
        self.val_S = state['val_S']
        self.test_S = state['test_S']
        self.hx = state['hx']
        self.hy = state['hy']
        self.median = state['median']
        self.random_diffusing_filters()
        self.create_samples_X_Y()

    def random_diffusing_filters(self):
        """
        Create two lineal random diffusing filters with L random coefficients
        using the graphs shift operators from Gx and Gy.
        Arguments:
            - L: number of filter coeffcients
        """
        self.Hx = np.zeros(self.Gx.W.shape)
        self.Hy = np.zeros(self.Gy.W.shape)
        Sx = self.Gx.W.todense()
        Sy = self.Gy.W.todense()
        for l in range(self.hx.size):
            self.Hx += self.hx[l]*np.linalg.matrix_power(Sx, l)
            self.Hy += self.hy[l]*np.linalg.matrix_power(Sy, l)

    def create_samples_X_Y(self):
        self.train_X = self.Hx.dot(self.train_S.T).T
        self.train_Y = self.Hy.dot(self.train_S.T).T
        self.val_X = self.Hx.dot(self.val_S.T).T
        self.val_Y = self.Hy.dot(self.val_S.T).T
        self.test_X = self.Hx.dot(self.test_S.T).T
        self.test_Y = self.Hy.dot(self.test_S.T).T
        if self.median:
            self.train_X = self.median_neighbours_nodes(self.train_X, self.Gx)
            self.train_Y = self.median_neighbours_nodes(self.train_Y, self.Gy)
            self.val_X = self.median_neighbours_nodes(self.val_X, self.Gx)
            self.val_Y = self.median_neighbours_nodes(self.val_Y, self.Gy)
            self.test_X = self.median_neighbours_nodes(self.test_X, self.Gx)
            self.test_Y = self.median_neighbours_nodes(self.test_Y, self.Gy)


class NonLinearDS2GS(DiffusedSparse2GS):
    def __init__(self, Gx, Gy, n_samples, L, n_delts, min_d=-1,
                 max_d=1, median=False, same_coeffs=False):
        super(NonLinearDS2GS, self).__init__(Gx, Gy, n_samples, L, n_delts,
                                             min_d, max_d)

        # TODO: check that linear is still a poor approximation for this
        h_x = np.random.rand(L)
        h_y = h_x if same_coeffs else np.random.rand(L)
        self.train_S = self.sparse_S(self.n_train, n_delts, min_d, max_d)
        self.val_S = self.sparse_S(self.n_val, n_delts, min_d, max_d)
        self.test_S = self.sparse_S(self.n_test, n_delts, min_d, max_d)
        self.train_X = self.signal_diffusion(self.train_S, self.Gx, h_x)
        self.train_Y = self.signal_diffusion(self.train_S, self.Gy, h_y)
        self.val_X = self.signal_diffusion(self.val_S, self.Gx, h_x)
        self.val_Y = self.signal_diffusion(self.val_S, self.Gy, h_y)
        self.test_X = self.signal_diffusion(self.test_S, self.Gx, h_x)
        self.test_Y = self.signal_diffusion(self.test_S, self.Gy, h_y)

        if median:
            self.train_X = self.median_neighbours_nodes(self.train_X, self.Gx)
            self.train_Y = self.median_neighbours_nodes(self.train_Y, self.Gy)
            self.val_X = self.median_neighbours_nodes(self.val_X, self.Gx)
            self.val_Y = self.median_neighbours_nodes(self.val_Y, self.Gy)
            self.test_X = self.median_neighbours_nodes(self.test_X, self.Gx)
            self.test_Y = self.median_neighbours_nodes(self.test_Y, self.Gy)

    def signal_diffusion(self, S, G, hs):
        S_T = S.T
        X_T = np.zeros(S_T.shape)
        A = G.W.todense()
        for l, h in enumerate(hs):
            X_T += h*np.linalg.matrix_power(A, l).dot(np.sign(S_T)*S_T**l)
        return X_T.T
