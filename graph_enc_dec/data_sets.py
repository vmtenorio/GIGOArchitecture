import numpy as np
import matplotlib.pyplot as plt
from pygsp.graphs import Graph, StochasticBlockModel, ErdosRenyi, BarabasiAlbert
from torch import Tensor

# Graph Type Constants
SBM = 1
ER = 2
BA = 3

# Comm Node Assignment Constants
CONT = 1    # Contiguous nodes
ALT = 2    # Alternated nodes
RAND = 3    # Random nodes


# Graph Relations
LINK_PERT = 1    # Perturbed links
NODE_PERT = 2    # Perturbed nodes
SBMS = 3    # 2 random SBMs
CLUSTER = 4    # Clusterized version


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
        G = StochasticBlockModel(N=ps['N'], k=ps['k'], p=ps['p'], z=z,
                                 q=ps['q'], connected=True, seed=seed,
                                 max_iter=25)
        G.set_coordinates('community2D')
        return G
    elif ps['type'] == ER:
        G = ErdosRenyi(N=ps['N'], p=ps['p'], connected=True, seed=seed,
                       max_iter=25)
        G.set_coordinates('community2D')
        return G
    elif ps['type'] == BA:
        G = BarabasiAlbert(N=ps['N'], m=ps['m'], m0=ps['m0'], seed=seed)
        G.info = {'comm_sizes': np.array([ps['N']]),
                  'node_com': np.zeros((ps['N'],), dtype=int)}
        G.set_coordinates('spring')
        return G
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
    links_c = np.random.choice(no_link_i[0].size, int(Gx.Ne * creat/100),
                               replace=False)
    idx_c = (no_link_i[0][links_c], no_link_i[1][links_c])

    # Destroy links
    link_i = np.where(A_x_triu == 1)
    links_d = np.random.choice(link_i[0].size, int(Gx.Ne * destr/100),
                               replace=False)
    idx_d = (link_i[0][links_d], link_i[1][links_d])

    A_x_triu[np.tril_indices(Gx.N)] = 0
    A_x_triu[idx_c] = 1
    A_x_triu[idx_d] = 0
    A_y = A_x_triu + A_x_triu.T
    return A_y


def perm_graph(A, coords, node_com, comm_sizes):
    N = A.shape[0]
    # Create permutation matrix
    P = np.zeros(A.shape)
    i = np.arange(N)
    j = np.random.permutation(N)
    P[i, j] = 1

    # Permute
    A_p = P.dot(A).dot(P.T)
    assert np.sum(np.diag(A_p)) == 0, 'Diagonal of permutated A is not 0'
    coords_p = P.dot(coords)
    node_com_p = P.dot(node_com)
    G = Graph(A_p)
    G.set_coordinates(coords_p)
    G.info = {'node_com': node_com_p,
              'comm_sizes': comm_sizes,
              'perm_matrix': P}
    return G


def perturbated_graphs(g_params, creat=5, dest=5, pct=True, perm=False, seed=None):
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
        Ay = perturbate_percentage(Gx, creat, dest)
    else:
        Ay = perturbate_probability(Gx, creat, dest)
    coords_Gy = Gx.coords
    node_com_Gy = Gx.info['node_com']
    comm_sizes_Gy = Gx.info['comm_sizes']
    assert np.sum(np.diag(Ay)) == 0, 'Diagonal of A is not 0'

    if perm:
        Gy = perm_graph(Ay, coords_Gy, node_com_Gy, comm_sizes_Gy)
    else:
        Gy = Graph(Ay)
        Gy.set_coordinates(coords_Gy)
        Gy.info = {'node_com': node_com_Gy,
                   'comm_sizes': comm_sizes_Gy}
    assert Gy.is_connected(), 'Could not create connected graph Gy'
    return Gx, Gy


def nodes_perturbated_graphs(g_params, n_dest, seed=None):
    Gx = create_graph(g_params, seed)
    Ax = Gx.A.todense()
    rm_nodes = np.random.choice(Gx.N, n_dest, replace=False)
    Ay = np.delete(Ax, rm_nodes, axis=0)
    Ay = np.delete(Ay, rm_nodes, axis=1)

    # Set Graph info
    coords_Gy = np.delete(Gx.coords, rm_nodes, axis=0)
    node_com_Gy = np.delete(Gx.info['node_com'], rm_nodes)
    comm_sizes_Gy = np.zeros(len(Gx.info['comm_sizes']))
    for i in range(len(Gx.info['comm_sizes'])):
        comm_sizes_Gy[i] = np.sum(node_com_Gy == i)

    Gy = Graph(Ay)
    assert Gy.is_connected(), 'Could not create connected graph Gy'
    Gy.set_coordinates(coords_Gy)
    Gy.info = {'node_com': node_com_Gy,
               'comm_sizes': comm_sizes_Gy,
               'rm_nodes': rm_nodes}
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
        if Gx.info['comm_sizes'].size != Gy.info['comm_sizes'].size:
            raise RuntimeError('Both graphs must have the same number of communities')

        if isinstance(n_samples, int):
            self.n_train = n_samples
            self.n_val = int(np.floor(0.25*n_samples))
            self.n_test = int(np.floor(0.25*n_samples))
        elif len(n_samples) == 3:
            self.n_train = n_samples[0]
            self.n_val = n_samples[1]
            self.n_test = n_samples[2]
        else:
            raise RuntimeError('n_samples must be an integer or a list with the \
                                samples for training, validation and test')
        self.Gx = Gx
        self.Gy = Gy

    def state_dict(self):
        state = {}
        state['train_Sx'] = self.train_Sx
        state['val_Sx'] = self.val_Sx
        state['test_Sx'] = self.test_Sx
        state['train_Sy'] = self.train_Sy
        state['val_Sy'] = self.val_Sy
        state['test_Sy'] = self.test_Sy
        state['hx'] = self.hx
        state['hy'] = self.hy
        state['median'] = self.median
        return state

    def load_state_dict(self, state, unit_norm=False):
        self.train_Sx = state['train_Sx']
        self.val_Sx = state['val_Sx']
        self.test_Sx = state['test_Sx']
        self.train_Sy = state['train_Sy']
        self.val_Sy = state['val_Sy']
        self.test_Sy = state['test_Sy']
        self.hx = state['hx']
        self.hy = state['hy']
        self.median = state['median']
        self.random_diffusing_filters()
        self.create_samples_X_Y()
        if unit_norm:
            self.to_unit_norm()

    def median_neighbours_nodes(self, X, G):
        X_aux = np.zeros(X.shape)
        for i in range(G.N):
            _, neighbours = np.asarray(G.W.todense()[i, :] != 0).nonzero()
            X_aux[:, i] = np.median(X[:, np.append(neighbours, i)], axis=1)
        return X_aux

    def to_tensor(self, n_chans=1):
        Nx = self.train_X.shape[1]
        Ny = self.train_Y.shape[1]
        self.train_X = Tensor(self.train_X).view([self.n_train, n_chans, Nx])
        self.train_Y = Tensor(self.train_Y).view([self.n_train, n_chans, Ny])
        self.val_X = Tensor(self.val_X).view([self.n_val, n_chans, Nx])
        self.val_Y = Tensor(self.val_Y).view([self.n_val, n_chans, Ny])
        self.test_X = Tensor(self.test_X).view([self.n_test, n_chans, Nx])
        self.test_Y = Tensor(self.test_Y).view([self.n_test, n_chans, Ny])

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
            Sx = self.train_Sx[id, :]
            Sy = self.train_Sy[id, :]
            X = self.train_X[id, :]
            Y = self.train_Y[id, :]
            _, axes = plt.subplots(2, 2)
            self.Gx.plot_signal(Sx, ax=axes[0, 0])
            self.Gx.plot_signal(X, ax=axes[0, 1])
            self.Gy.plot_signal(Sy, ax=axes[1, 0])
            self.Gy.plot_signal(Y, ax=axes[1, 1])
        if show:
            plt.show()

    def delta_values(self, G, n_samp, n_deltas, min_delta, max_delta):
        n_comms = G.info['comm_sizes'].size
        if n_comms > 1:
            step = (max_delta-min_delta)/(n_comms-1)
        else:
            step = (max_delta-min_delta)/(n_deltas-1)
        ds_per_comm = np.ceil(n_deltas/n_comms).astype(int)
        delta_means = np.arange(min_delta, max_delta+0.1, step)
        delta_means = np.tile(delta_means, ds_per_comm)[:n_deltas]
        delt_val = np.zeros((n_deltas, n_samp))
        for i in range(n_samp):
            delt_val[:, i] = np.random.randn(n_deltas)*step/4 + delta_means
        return delt_val

    def sparse_S(self, G, delta_values):
        """
        Create random sparse signal s composed of different deltas placed in the
        different communities of the graph. If the graph is an ER, then deltas
        are just placed on random nodes
        """
        n_samp = delta_values.shape[1]
        S = np.zeros((G.N, n_samp))
        # Randomly assign delta value to comm nodes
        for i in range(n_samp):
            for j in range(delta_values.shape[0]):
                delta = delta_values[j, i]
                com_j = j % G.info['comm_sizes'].size
                com_nodes, = np.asarray(G.info['node_com'] == com_j).nonzero()
                rand_index = np.random.randint(0, G.info['comm_sizes'][com_j])
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
    '''
    With this class X and Y will be transformations of an sparse intermediate space Sx and Sy.
    However, despite Sx and Sy having the same nonzero values, these deltas will be placed
    on different nodes (but on the same community) even  if Gx and Gy are equal. For assigning
    deltas to the exact same nodes please use the class LinearDS2GSLinksPert.
    '''
    def __init__(self, Gx, Gy, n_samples, L, n_delts, min_d=-1,
                 max_d=1, median=True, same_coeffs=False, neg_coeffs=False):
        super(LinearDS2GS, self).__init__(Gx, Gy, n_samples, L, n_delts, min_d,
                                          max_d)
        self.median = median
        if neg_coeffs:
            self.hx = 2 * np.random.rand(L) - 1
            self.hy = self.hx if same_coeffs else (2 * np.random.rand(L) - 1)
        else:
            self.hx = np.random.rand(L)
            self.hy = self.hx if same_coeffs else np.random.rand(L)
        self.random_diffusing_filters()
        self.create_samples_S(n_delts, min_d, max_d)
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

    def create_samples_S(self, delts, min_d, max_d):
        train_deltas = self.delta_values(self.Gx, self.n_train, delts, min_d, max_d)
        val_deltas = self.delta_values(self.Gx, self.n_val, delts, min_d, max_d)
        test_deltas = self.delta_values(self.Gx, self.n_test, delts, min_d, max_d)
        self.train_Sx = self.sparse_S(self.Gx, train_deltas)
        self.val_Sx = self.sparse_S(self.Gx, val_deltas)
        self.test_Sx = self.sparse_S(self.Gx, test_deltas)
        self.train_Sy = self.sparse_S(self.Gy, train_deltas)
        self.val_Sy = self.sparse_S(self.Gy, val_deltas)
        self.test_Sy = self.sparse_S(self.Gy, test_deltas)

    def create_samples_X_Y(self):
        self.train_X = self.Hx.dot(self.train_Sx.T).T
        self.train_Y = self.Hy.dot(self.train_Sy.T).T
        self.val_X = self.Hx.dot(self.val_Sx.T).T
        self.val_Y = self.Hy.dot(self.val_Sy.T).T
        self.test_X = self.Hx.dot(self.test_Sx.T).T
        self.test_Y = self.Hy.dot(self.test_Sy.T).T
        if self.median:
            self.train_X = self.median_neighbours_nodes(self.train_X, self.Gx)
            self.train_Y = self.median_neighbours_nodes(self.train_Y, self.Gy)
            self.val_X = self.median_neighbours_nodes(self.val_X, self.Gx)
            self.val_Y = self.median_neighbours_nodes(self.val_Y, self.Gy)
            self.test_X = self.median_neighbours_nodes(self.test_X, self.Gx)
            self.test_Y = self.median_neighbours_nodes(self.test_Y, self.Gy)


class LinearDS2GSLinksPert(LinearDS2GS):
    def __init__(self, Gx, Gy, n_samples, L, n_delts, min_d=-1,
                 max_d=1, median=True, same_coeffs=False):
        assert Gx.N == Gy.N, 'Graphs Gx and Gy must have the same size'
        super(LinearDS2GSLinksPert, self).__init__(Gx, Gy, n_samples, L,
                                                   n_delts, min_d, max_d)

    def create_samples_S(self, delts, min_d, max_d):
        train_deltas = self.delta_values(self.Gx, self.n_train, delts, min_d, max_d)
        val_deltas = self.delta_values(self.Gx, self.n_val, delts, min_d, max_d)
        test_deltas = self.delta_values(self.Gx, self.n_test, delts, min_d, max_d)
        self.train_Sx = self.sparse_S(self.Gx, train_deltas)
        self.val_Sx = self.sparse_S(self.Gx, val_deltas)
        self.test_Sx = self.sparse_S(self.Gx, test_deltas)
        if 'perm_matrix' in self.Gy.info.keys():
            self.train_Sy = self.train_Sx.dot(self.Gy.info['perm_matrix'].T)
            self.val_Sy = self.val_Sx.dot(self.Gy.info['perm_matrix'].T)
            self.test_Sy = self.test_Sx.dot(self.Gy.info['perm_matrix'].T)
        else:
            self.train_Sy = np.copy(self.train_Sx)
            self.val_Sy = np.copy(self.val_Sx)
            self.test_Sy = np.copy(self.test_Sx)


class LinearDS2GSNodesPert(LinearDS2GS):
    def __init__(self, Gx, Gy, n_samples, L, n_delts, min_d=-1,
                 max_d=1, median=True, same_coeffs=False):
        super(LinearDS2GSNodesPert, self).__init__(Gx, Gy, n_samples, L,
                                                   n_delts, min_d, max_d)

    def create_samples_S(self, delts, min_d, max_d):
        train_deltas = self.delta_values(self.Gx, self.n_train, delts, min_d, max_d)
        val_deltas = self.delta_values(self.Gx, self.n_val, delts, min_d, max_d)
        test_deltas = self.delta_values(self.Gx, self.n_test, delts, min_d, max_d)
        self.train_Sx, self.train_Sy = self.sparse_S(self.Gx, train_deltas)
        self.val_Sx, self.val_Sy = self.sparse_S(self.Gx, val_deltas)
        self.test_Sx, self.test_Sy = self.sparse_S(self.Gx, test_deltas)

    def sparse_S(self, G, delta_values):
        n_samp = delta_values.shape[1]
        Sx = np.zeros((G.N, n_samp))
        Sy = np.zeros((G.N, n_samp))
        rm_nodes = self.Gy.info['rm_nodes']
        for i in range(n_samp):
            for j in range(delta_values.shape[0]):
                delta = delta_values[j, i]
                com_j = j % G.info['comm_sizes'].size
                com_nodes, = np.asarray(G.info['node_com'] == com_j).nonzero()
                for n in rm_nodes:
                    com_nodes = np.delete(com_nodes, np.where(com_nodes == n))
                rand_index = np.random.randint(0, len(com_nodes))
                Sx[com_nodes[rand_index], i] = delta
                Sy[com_nodes[rand_index], i] = delta
        Sy = np.delete(Sy, rm_nodes, axis=0)
        return Sx.T, Sy.T


class NonLinearDS2GS(DiffusedSparse2GS):
    def __init__(self, Gx, Gy, n_samples, L, n_delts, min_d=-1,
                 max_d=1, median=False, same_coeffs=False, neg_coeffs=False):
        super(NonLinearDS2GS, self).__init__(Gx, Gy, n_samples, L, n_delts,
                                             min_d, max_d)

        # TODO: check that linear is still a poor approximation for this
        if neg_coeffs:
            self.hx = 2 * np.random.rand(L) - 1
            self.hy = self.hx if same_coeffs else (2 * np.random.rand(L) - 1)
        else:
            self.hx = np.random.rand(L)
            self.hy = self.hx if same_coeffs else np.random.rand(L)
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
