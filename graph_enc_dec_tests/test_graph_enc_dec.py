import unittest
import sys
import numpy as np
from torch.nn import Sequential
from torch import Tensor, zeros

sys.path.insert(0, '.')
sys.path.insert(0, '.\graph_enc_dec')
from graph_enc_dec import data_sets as ds
from graph_enc_dec import graph_clustering as gc
from graph_enc_dec.architecture import GraphEncoderDecoder, GraphDownsampling, GraphUpsampling
from graph_enc_dec.model import Model

SEED = 15


# TODO: test graph creation --> check graphs are symm
class LinkPerturbatedGraphsTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        self.G_params = {}
        self.G_params['type'] = ds.SBM  # SBM, ER or BA
        self.G_params['N'] = self.N = 256
        self.G_params['k'] = k = 4
        self.G_params['p'] = 0.20
        self.G_params['q'] = 0.015/4
        self.G_params['type_z'] = ds.RAND

    def test_graph_reproducibility(self):
        Gx, Gy = ds.perturbated_graphs(self.G_params, 10,
                                       10, pct=True, seed=SEED)
        print('Link x:', Gx.Ne)
        print('Link y:', Gy.Ne)
        self.assertEqual(1664, Gx.Ne)
        self.assertEqual(1664, Gy.Ne)
        print('Diff links:', np.sum(Gx.A != Gy.A)/2/Gx.Ne)
        self.assertAlmostEqual(0.19951923076923078,
                               np.sum(Gx.A != Gy.A)/2/Gx.Ne)

    def test_probability_perturbation(self):
        create = 0.0005
        destroy = 0.05
        n_graphs = 10
        diff_links = np.zeros(n_graphs)
        exp_err = np.zeros(n_graphs)
        margin = 6
        for i in range(n_graphs):
            Gx, Gy = ds.perturbated_graphs(self.G_params, create,
                                           destroy, pct=False, seed=SEED)
            Ax = Gx.W.todense()
            Ay = Gy.W.todense()
            self.assertFalse(Gx.is_directed())
            self.assertTrue(Gx.is_connected())
            self.assertEqual(np.sum(np.diag(Ax)), 0)
            self.assertFalse(Gy.is_directed())
            self.assertTrue(Gy.is_connected())
            self.assertEqual(np.sum(np.diag(Ay)), 0)
            diff_links[i] = np.sum(Ax != Ay)/Gx.Ne/2
            exp_err[i] = (create*(Gx.N*(Gx.N-1)/2-Gx.Ne) + Gx.Ne*destroy)/Gx.N
        self.assertTrue(np.mean(diff_links) <= (np.mean(exp_err)+margin)/100)
        self.assertTrue(np.mean(diff_links) >= (np.mean(exp_err)-margin)/100)

    def test_percentage_perturbation(self):
        create = destroy = 5
        up_err_margin = (create + destroy + 2)/100
        bottom_err_margin = (create + destroy - 2)/100
        for i in range(10):
            Gx, Gy = ds.perturbated_graphs(self.G_params, create,
                                           destroy, pct=True, seed=SEED)
            Ax = Gx.W.todense()
            Ay = Gy.W.todense()
            print('diff:', np.sum(Ax != Ay)/Gx.Ne/2)
            self.assertFalse(Gx.is_directed())
            self.assertTrue(Gx.is_connected())
            self.assertEqual(np.sum(np.diag(Ax)), 0)
            self.assertFalse(Gy.is_directed())
            self.assertTrue(Gy.is_connected())
            self.assertEqual(np.sum(np.diag(Ay)), 0)
            self.assertTrue(np.sum(Ax != Ay)/Gx.Ne/2 <= up_err_margin)
            self.assertTrue(np.sum(Ax != Ay)/Gx.Ne/2 >= bottom_err_margin)

    def test_permute_graph(self):
        Gx, Gy = ds.perturbated_graphs(self.G_params, 0, 0,
                                       perm=True, pct=True, seed=SEED)
        Ax = Gx.W.todense()
        Ay = Gy.W.todense()
        comm_X = Gx.info['node_com']
        comm_Y = Gy.info['node_com']
        P = Gy.info['perm_matrix']
        self.assertFalse(np.array_equal(Ax, Ay))
        self.assertFalse(np.array_equal(comm_X, comm_Y))
        self.assertTrue(np.array_equal(np.eye(Gx.N), P.dot(P.T)))
        self.assertTrue(np.array_equal(Ax, P.T.dot(Ay).dot(P)))
        self.assertTrue(np.array_equal(comm_X, P.T.dot(comm_Y)))


class NodePerturbatedGraphsTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        self.G_params = {}
        self.G_params['type'] = ds.SBM  # SBM, ER or BA
        self.G_params['N'] = self.N = 256
        self.G_params['k'] = k = 4
        self.G_params['p'] = 0.20
        self.G_params['q'] = 0.015/4
        self.G_params['type_z'] = ds.RAND

    def test_perturbation(self):
        dest = 30
        for i in range(10):
            Gx, Gy = ds.nodes_perturbated_graphs(self.G_params, dest,
                                                 seed=SEED)
            Ax = Gx.W.todense()
            Ay = Gy.W.todense()
            Ax_rm = np.delete(Ax, Gy.info['rm_nodes'], axis=0)
            Ax_rm = np.delete(Ax_rm, Gy.info['rm_nodes'], axis=1)
            rm_comms = np.delete(Gx.info['node_com'], Gy.info['rm_nodes'])
            self.assertFalse(Gx.is_directed())
            self.assertTrue(Gx.is_connected())
            self.assertEqual(np.sum(np.diag(Ax)), 0)
            self.assertFalse(Gy.is_directed())
            self.assertTrue(Gy.is_connected())
            self.assertEqual(np.sum(np.diag(Ay)), 0)
            self.assertEqual(Gy.N, Gx.N-dest)
            self.assertTrue(np.array_equal(Ax_rm, Ay))
            self.assertTrue(np.array_equal(rm_comms, Gy.info['node_com']))

    def test_permute_graph(self):
        dest = 30
        Gx, Gy = ds.nodes_perturbated_graphs(self.G_params, dest,
                                             perm=True, seed=SEED)
        Ax = Gx.W.todense()
        Ay = Gy.W.todense()
        P = Gy.info['perm_matrix']
        Ax_rm = np.delete(Ax, Gy.info['rm_nodes'], axis=0)
        Ax_rm = np.delete(Ax_rm, Gy.info['rm_nodes'], axis=1)
        rm_comms_X = np.delete(Gx.info['node_com'], Gy.info['rm_nodes'])
        comm_Y = Gy.info['node_com']
        self.assertFalse(np.array_equal(Ax_rm, Ay))
        self.assertFalse(np.array_equal(rm_comms_X, comm_Y))
        self.assertTrue(np.array_equal(np.eye(Gy.N), P.dot(P.T)))
        self.assertTrue(np.array_equal(Ax_rm, P.T.dot(Ay).dot(P)))
        self.assertTrue(np.array_equal(rm_comms_X, P.T.dot(comm_Y)))


class LinearDS2GS2GSTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        self.G_params = {}
        self.G_params['type'] = ds.SBM
        self.G_params['N'] = 32
        self.G_params['k'] = 4
        self.G_params['p'] = 0.7
        self.G_params['q'] = 0.1
        self.G_params['type_z'] = ds.RAND
        self.eps1 = 5
        self.eps2 = 5
        self.Gx, self.Gy = ds.perturbated_graphs(self.G_params, self.eps1,
                                                 self.eps2, seed=SEED)

    def test_S_ER(self):
        n_samps = [50, 20, 20]
        L = 6
        n_delts = 6
        self.G_params['type'] = ds.ER
        Gx, Gy = ds.perturbated_graphs(self.G_params, self.eps1, self.eps2, seed=SEED)
        data = ds.LinearDS2GS(Gx, Gy, n_samps, L, n_delts)
        data.to_unit_norm()
        self.assertFalse(np.array_equal(data.Hx, data.Hy))
        for i in range(n_samps[0]):
            self.assertLessEqual(np.sum(data.train_Sx[i,:][data.train_Sx[i,:]!=0]), n_delts)
            self.assertLessEqual(np.sum(data.train_Sy[i,:][data.train_Sy[i,:]!=0]), n_delts)
        for i in range(n_samps[1]):
            self.assertLessEqual(np.sum(data.train_Sx[i,:][data.train_Sx[i,:]!=0]), n_delts)
            self.assertLessEqual(np.sum(data.train_Sy[i,:][data.train_Sy[i,:]!=0]), n_delts)
        for i in range(n_samps[2]):
            self.assertLessEqual(np.sum(data.train_Sx[i,:][data.train_Sx[i,:]!=0]), n_delts)
            self.assertLessEqual(np.sum(data.train_Sy[i,:][data.train_Sy[i,:]!=0]), n_delts)

    # TODO: check if deltas are in the comms
    def test_S_SBM(self):
        n_samps = [50, 20, 20]
        L = 6
        n_delts = self.G_params['k']
        data = ds.LinearDS2GS(self.Gx, self.Gy, n_samps, L, n_delts)
        self.assertFalse(np.array_equal(data.Hx, data.Hy))
        for i in range(n_samps[0]):
            self.assertLessEqual(np.sum(data.train_Sx[i,:][data.train_Sx[i,:]!=0]), n_delts)
            self.assertLessEqual(np.sum(data.train_Sy[i,:][data.train_Sy[i,:]!=0]), n_delts)
        for i in range(n_samps[1]):
            self.assertLessEqual(np.sum(data.train_Sx[i,:][data.train_Sx[i,:]!=0]), n_delts)
            self.assertLessEqual(np.sum(data.train_Sy[i,:][data.train_Sy[i,:]!=0]), n_delts)
        for i in range(n_samps[2]):
            self.assertLessEqual(np.sum(data.train_Sx[i,:][data.train_Sx[i,:]!=0]), n_delts)
            self.assertLessEqual(np.sum(data.train_Sy[i,:][data.train_Sy[i,:]!=0]), n_delts)

        n_delts = self.G_params['k']*2+3
        data = ds.LinearDS2GS(self.Gx, self.Gy, n_samps, L, n_delts)
        for i in range(n_samps[0]):
            self.assertLessEqual(np.sum(data.train_Sx[i,:][data.train_Sx[i,:]!=0]), n_delts)
            self.assertLessEqual(np.sum(data.train_Sy[i,:][data.train_Sy[i,:]!=0]), n_delts)
        for i in range(n_samps[1]):
            self.assertLessEqual(np.sum(data.train_Sx[i,:][data.train_Sx[i,:]!=0]), n_delts)
            self.assertLessEqual(np.sum(data.train_Sy[i,:][data.train_Sy[i,:]!=0]), n_delts)
        for i in range(n_samps[2]):
            self.assertLessEqual(np.sum(data.train_Sx[i,:][data.train_Sx[i,:]!=0]), n_delts)
            self.assertLessEqual(np.sum(data.train_Sy[i,:][data.train_Sy[i,:]!=0]), n_delts)

    def test_state_dict(self):
        n_samps = [100, 50, 50]
        L = 6
        n_delts = self.G_params['k']
        # Test saving state
        data = ds.LinearDS2GS(self.Gx, self.Gy, n_samps, L, n_delts)
        state = data.state_dict()
        self.assertTrue(np.array_equal(data.train_Sx, state['train_Sx']))
        self.assertTrue(np.array_equal(data.val_Sy, state['val_Sy']))
        self.assertTrue(np.array_equal(data.test_Sy, state['test_Sy']))
        self.assertTrue(np.array_equal(data.hx, state['hx']))
        self.assertTrue(np.array_equal(data.hy, state['hy']))
        self.assertEqual(data.median, state['median'])

        # Test loading state
        data2 = ds.LinearDS2GS(self.Gx, self.Gy, n_samps, L, n_delts)
        self.assertFalse(np.array_equal(data2.train_Sx, data.train_Sx))
        self.assertFalse(np.array_equal(data2.val_Sx, data.val_Sx))
        self.assertFalse(np.array_equal(data2.test_Sx, data.test_Sx))
        self.assertFalse(np.array_equal(data2.train_X, data.train_X))
        self.assertFalse(np.array_equal(data2.val_X, data.val_X))
        self.assertFalse(np.array_equal(data2.test_X, data.test_X))
        self.assertFalse(np.array_equal(data2.train_Y, data.train_Y))
        self.assertFalse(np.array_equal(data2.val_Y, data.val_Y))
        self.assertFalse(np.array_equal(data2.test_Y, data.test_Y))
        data2.load_state_dict(state)
        self.assertTrue(np.array_equal(data2.train_X, data.train_X))
        self.assertTrue(np.array_equal(data2.val_X, data.val_X))
        self.assertTrue(np.array_equal(data2.test_X, data.test_X))
        self.assertTrue(np.array_equal(data2.train_Y, data.train_Y))
        self.assertTrue(np.array_equal(data2.val_Y, data.val_Y))
        self.assertTrue(np.array_equal(data2.test_Y, data.test_Y))

    def test_to_unit_norm(self):
        n_samps = [50, 20, 20]
        L = 6
        n_delts = 4
        data = ds.LinearDS2GS(self.Gx, self.Gy, n_samps, L, n_delts)
        data.to_unit_norm()

        for i in range(n_samps[0]):
            self.assertAlmostEqual(np.linalg.norm(data.train_X[i,:]),1)
            self.assertAlmostEqual(np.linalg.norm(data.train_Y[i,:]),1)
        for i in range(n_samps[1]):
            self.assertAlmostEqual(np.linalg.norm(data.val_X[i,:]),1)
            self.assertAlmostEqual(np.linalg.norm(data.val_Y[i,:]),1)
        for i in range(n_samps[2]):
            self.assertAlmostEqual(np.linalg.norm(data.test_X[i,:]),1)
            self.assertAlmostEqual(np.linalg.norm(data.train_Y[i,:]),1)


class LinearDS2GSLinksPertTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        self.G_params = {}
        self.G_params['type'] = ds.SBM
        self.G_params['N'] = 32
        self.G_params['k'] = 4
        self.G_params['p'] = 0.8
        self.G_params['q'] = 0.1
        self.G_params['type_z'] = ds.RAND
        self.eps1 = 5
        self.eps2 = 5
        self.Gx, self.Gy = ds.perturbated_graphs(self.G_params, self.eps1,
                                                 self.eps2, seed=SEED)

    def test_same_S(self):
        n_samps = [50, 20, 20]
        L = 6
        n_delts = self.G_params['k']
        data = ds.LinearDS2GSLinksPert(self.Gx, self.Gy, n_samps, L, n_delts)
        self.assertFalse(np.array_equal(data.Hx, data.Hy))
        self.assertTrue(np.array_equal(data.train_Sx, data.train_Sy))
        self.assertTrue(np.array_equal(data.val_Sx, data.val_Sy))
        self.assertTrue(np.array_equal(data.test_Sx, data.test_Sy))

        for i in range(n_samps[0]):
            self.assertLessEqual(np.sum(data.train_Sy[i,:][data.train_Sy[i,:]!=0]), n_delts)
        for i in range(n_samps[1]):
            self.assertLessEqual(np.sum(data.train_Sx[i,:][data.train_Sx[i,:]!=0]), n_delts)
        for i in range(n_samps[2]):
            self.assertLessEqual(np.sum(data.train_Sy[i,:][data.train_Sy[i,:]!=0]), n_delts)

    def test_permutated_S(self):
        n_samps = [50, 20, 20]
        L = 6
        n_delts = self.G_params['k']
        Gx, Gy = ds.perturbated_graphs(self.G_params, 0, 0,
                                       perm=True, seed=SEED)
        data = ds.LinearDS2GSLinksPert(Gx, Gy, n_samps, L, n_delts)
        P = data.Gy.info['perm_matrix']
        self.assertFalse(np.array_equal(data.Hx, data.Hy))
        self.assertFalse(np.array_equal(data.train_Sx, data.train_Sy))
        self.assertFalse(np.array_equal(data.val_Sx, data.val_Sy))
        self.assertFalse(np.array_equal(data.test_Sx, data.test_Sy))
        self.assertTrue(np.array_equal(data.train_Sx, data.train_Sy.dot(P)))
        self.assertTrue(np.array_equal(data.val_Sx, data.val_Sy.dot(P)))
        self.assertTrue(np.array_equal(data.test_Sx, data.test_Sy.dot(P)))


# TODO: test if a community is empty!
class LinearDS2GSNodesPertTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        self.G_params = {}
        self.G_params['type'] = ds.SBM
        self.G_params['N'] = 32
        self.G_params['k'] = 4
        self.G_params['p'] = 0.8
        self.G_params['q'] = 0.1
        self.G_params['type_z'] = ds.RAND
        self.dest = 5
        self.Gx, self.Gy = ds.nodes_perturbated_graphs(self.G_params, self.dest,
                                                       seed=SEED)

    def test_same_S(self):
        n_samps = [100, 20, 20]
        L = 6
        n_delts = self.G_params['k']
        rm_nodes = self.Gy.info['rm_nodes']
        data = ds.LinearDS2GSNodesPert(self.Gx, self.Gy, n_samps, L, n_delts)
        train_Sx_rm = np.delete(data.train_Sx, rm_nodes, axis=1)
        val_Sx_rm = np.delete(data.val_Sx, rm_nodes, axis=1)
        test_Sx_rm = np.delete(data.test_Sx, rm_nodes, axis=1)
        self.assertFalse(np.array_equal(data.Hx, data.Hy))
        self.assertTrue(np.array_equal(train_Sx_rm, data.train_Sy))
        self.assertTrue(np.array_equal(val_Sx_rm, data.val_Sy))
        self.assertTrue(np.array_equal(test_Sx_rm, data.test_Sy))
        for i in range(n_samps[0]):
            self.assertLessEqual(np.sum(data.train_Sy[i,:][data.train_Sy[i,:]!=0]), n_delts)
        for i in range(n_samps[1]):
            self.assertLessEqual(np.sum(data.train_Sx[i,:][data.train_Sx[i,:]!=0]), n_delts)
        for i in range(n_samps[2]):
            self.assertLessEqual(np.sum(data.train_Sy[i,:][data.train_Sy[i,:]!=0]), n_delts)

    def test_permutated_S(self):
        n_samps = [50, 20, 20]
        L = 6
        n_delts = self.G_params['k']
        Gx, Gy = ds.nodes_perturbated_graphs(self.G_params, self.dest,
                                             perm=True, seed=SEED)
        data = ds.LinearDS2GSNodesPert(Gx, Gy, n_samps, L, n_delts)
        P = data.Gy.info['perm_matrix']
        rm_nodes = Gy.info['rm_nodes']
        train_Sx_rm = np.delete(data.train_Sx, rm_nodes, axis=1)
        val_Sx_rm = np.delete(data.val_Sx, rm_nodes, axis=1)
        test_Sx_rm = np.delete(data.test_Sx, rm_nodes, axis=1)
        self.assertFalse(np.array_equal(train_Sx_rm, data.train_Sy))
        self.assertFalse(np.array_equal(val_Sx_rm, data.val_Sy))
        self.assertFalse(np.array_equal(test_Sx_rm, data.test_Sy))
        self.assertTrue(np.array_equal(train_Sx_rm, data.train_Sy.dot(P)))
        self.assertTrue(np.array_equal(val_Sx_rm, data.val_Sy.dot(P)))
        self.assertTrue(np.array_equal(test_Sx_rm, data.test_Sy.dot(P)))


class GraphClustSizesTest(unittest.TestCase):
    def setUp(self):
        self.G_params = {}
        self.G_params['type'] = ds.SBM
        self.G_params['N'] = 256
        self.G_params['k'] = 4
        self.G_params['p'] = 0.15
        self.G_params['q'] = 0.01/4
        self.G_params['type_z'] = ds.CONT
        self.G = ds.create_graph(self.G_params, seed=SEED)

    def test_wrong_sizes_enc(self):
        nodes = [256, 256, 64, 64, 32, 100, 16, 4, 4]
        try:
            gc.MultiResGraphClustering(self.G, nodes, k=4, up_method=None)
            self.fail()
        except:
            pass

    def all_sizes_diff_enc(self):
        nodes = [256, 64, 32, 16, 4]
        cluster = gc.MultiResGraphClustering(self.G, nodes, k=4, up_method=None)
        self.assertEqual(len(nodes), len(cluster.sizes))
        self.assertEqual(len(nodes)-1, len(cluster.Ds))

    def repeated_sizes_enc(self):
        nodes = [256, 256, 256, 64, 32, 32, 16, 4, 4]
        no_rep_nodes = set(nodes)
        cluster = gc.MultiResGraphClustering(self.G, nodes, k=4, up_method=None)
        self.assertEqual(len(nodes), len(cluster.sizes))
        self.assertEqual(len(no_rep_nodes)-1, len(cluster.Ds))

    def test_wrong_sizes_dec(self):
        nodes = [4, 4, 16, 32, 32, 64, 64, 256, 64, 256, 256]
        try:
            gc.MultiResGraphClustering(self.G, nodes, k=4, up_method=None)
            self.fail()
        except:
            pass

    def all_sizes_diff_dec(self):
        nodes = [4,16,32,64,256]
        cluster = gc.MultiResGraphClustering(self.G, nodes, k=4, up_method=None)
        self.assertEqual(len(nodes), len(cluster.sizes))
        self.assertEqual(len(nodes)-1, len(cluster.Us))

    def repeated_sizes_dec(self):
        nodes = [4,4,16,32,32,64,256,256,256]
        no_rep_nodes = set(nodes)
        cluster = gc.MultiResGraphClustering(self.G, nodes, k=4, up_method=None)
        self.assertEqual(len(nodes), len(cluster.sizes))
        self.assertEqual(len(no_rep_nodes)-1, len(cluster.Us))

    def all_sizes_repeated(self):
        nodes = [256,256,256]
        cluster = gc.MultiResGraphClustering(self.G, nodes, k=4, up_method=None)
        self.assertEqual(len(nodes), len(cluster.sizes))
        self.assertEqual(len([]),len(cluster.Ds))

class GraphClustMatricesTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        self.G_params = {}
        self.G_params['type'] = ds.SBM
        self.G_params['N']  = 256
        self.G_params['k']  = 4
        self.G_params['p'] = 0.15
        self.G_params['q'] = 0.01/4

    def test_cluster_sizes_dec(self):
        self.G_params['type_z'] = ds.CONT
        G = ds.create_graph(self.G_params, seed=SEED)

    def test_downsampling_matrices(self):
        nodes_enc = [256,64,32,16,4]
        ups = gc.WEI

        # Contiguous nodes
        self.G_params['type_z'] = ds.CONT
        G = ds.create_graph(self.G_params, seed=SEED)
        cluster = gc.MultiResGraphClustering(G, nodes_enc, k=4, up_method=ups)
        out = np.array(cluster.labels[-1])
        for D in cluster.Ds:
            out = D.dot(out)/np.sum(D,1)
        self.assertTrue(np.array_equal(out, np.unique(cluster.labels[-1])))

        # Alternated nodes
        self.G_params['type_z'] = ds.ALT
        G = ds.create_graph(self.G_params, seed=SEED)
        cluster = gc.MultiResGraphClustering(G, nodes_enc, k=4, up_method=ups)
        out = np.array(cluster.labels[-1])
        for D in cluster.Ds:
            out = D.dot(out)/np.sum(D,1)
        self.assertTrue(np.array_equal(out, np.unique(cluster.labels[-1])))

        # Random nodes
        self.G_params['type_z'] = ds.RAND
        G = ds.create_graph(self.G_params, seed=SEED)
        cluster = gc.MultiResGraphClustering(G, nodes_enc, k=4, up_method=ups)
        out = np.array(cluster.labels[-1])
        for D in cluster.Ds:
            out = D.dot(out)/np.sum(D,1)
        self.assertTrue(np.array_equal(out, np.unique(cluster.labels[-1])))

    def test_upsampling_matrices(self):
        nodes_enc = [4,16,32,64,256]
        ups = gc.WEI

        # Contiguous nodes
        self.G_params['type_z'] = ds.CONT
        G = ds.create_graph(self.G_params, seed=SEED)
        cluster = gc.MultiResGraphClustering(G, nodes_enc, k=4, up_method=ups)
        out = np.unique(cluster.labels[0])
        for U in cluster.Us:
            out = U.dot(out)
        self.assertTrue(np.array_equal(out,cluster.labels[0]))

        # Alternated nodes
        self.G_params['type_z'] = ds.ALT
        G = ds.create_graph(self.G_params, seed=SEED) 
        cluster = gc.MultiResGraphClustering(G, nodes_enc, k=4, up_method=ups)
        out = np.unique(cluster.labels[0])
        for U in cluster.Us:
            out = U.dot(out)
        self.assertTrue(np.array_equal(out,cluster.labels[0]))

        # Random nodes
        self.G_params['type_z'] = ds.RAND
        G = ds.create_graph(self.G_params, seed=SEED)
        cluster = gc.MultiResGraphClustering(G, nodes_enc, k=4, up_method=ups)
        out = np.unique(cluster.labels[0])
        for U in cluster.Us:
            out = U.dot(out)
        self.assertTrue(np.array_equal(out,cluster.labels[0]))

class BuildNetworkTest(unittest.TestCase):
    def test_wrong_inputs(self):
        fts_enc = [1, 2, 4]
        n_enc = [10, 5, 2]
        fts_dec = [3, 2, 2] 
        n_dec = [2, 5, 10]
        fts_conv = [2, 2, 1] 
        try:
            GraphEncoderDecoder(fts_enc,n_enc,[],fts_dec,n_dec,[],fts_conv)
            self.fail()
        except:
            pass
        fts_enc = [1, 2, 3]
        n_dec = [3, 5, 10]
        try:
            GraphEncoderDecoder(fts_enc,n_enc,[],fts_dec,n_dec,[],fts_conv)
            self.fail()
        except:
            pass
        n_dec = [2, 5, 10]
        fts_dec = [3, 2, 1]
        try:
            GraphEncoderDecoder(fts_enc,n_enc,[],fts_dec,n_dec,[],fts_conv)
            self.fail()
        except:
            pass
        fts_dec = [3, 2, 2, 1]
        try:
            GraphEncoderDecoder(fts_enc,n_enc,[],fts_dec,n_dec,[],fts_conv)
            self.fail()
        except:
            pass
        fts_dec = [3, 2, 1]
        n_enc = [10, 5, 5, 2]
        try:
            GraphEncoderDecoder(fts_enc,n_enc,[],fts_dec,n_dec,[],fts_conv)
            self.fail()
        except:
            pass
        
    def test_complete_network(self):
        pass

    def test_missing_upsamplings(self):
        pass

    def test_missing_downsamplings(self):
        pass

    def test_missing_ups_and_down(self):
        pass

    def test_missing_conv_section(self):
        pass

    # TODO: URGENT: test UPSM and DOWNS modules

class GraphDownsamplingTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        G_params = {}
        G_params['type'] = ds.SBM
        G_params['N']  = 256
        G_params['k']  = 4
        G_params['p'] = 0.15
        G_params['q'] = 0.01/4
        G_params['type_z'] = ds.RAND
        G = ds.create_graph(G_params, seed=SEED)
        nodes_enc = [256,64,32,16,4]
        ups = gc.WEI
        self.N = nodes_enc[0]
        self.k = nodes_enc[-1]
        self.cluster = gc.MultiResGraphClustering(G, nodes_enc, k=4, up_method=ups)
        self.model = Sequential()
        for i, D in enumerate(self.cluster.Ds):
            self.add_layer(GraphDownsampling(D, self.cluster.As[i+1], gamma=1))


    def add_layer(self, module):
        self.model.add_module(str(len(self.model) + 1), module)

    def test_1D_downsampling(self):
        expected_result = np.unique(self.cluster.labels[-1]).astype(np.float32)
        expected_result = Tensor(expected_result).view([1,1,self.k])
        input = Tensor(self.cluster.labels[-1].astype(np.float32)).view([1, 1, self.N])
        result = self.model(input)
        self.assertTrue(result.equal(expected_result))

    def test_downsampling_with_channels(self):
        n_chans = 3
        expected_result = np.unique([self.cluster.labels[-1]],axis=1).astype(np.float32)
        expected_result = np.repeat(expected_result, n_chans, axis=0)
        expected_result = Tensor(expected_result).view([1, n_chans, self.k])
        input = np.repeat([self.cluster.labels[-1]], n_chans, axis=0).astype(np.float32)
        input = Tensor(input).view([1, n_chans, self.N])
        result = self.model(input)
        self.assertTrue(result.equal(expected_result))

    def test_downsampling_with_samples(self):
        n_samples = 5
        expected_result = np.unique([self.cluster.labels[-1]],axis=1).astype(np.float32)
        expected_result = np.repeat(expected_result, n_samples, axis=0)
        expected_result = Tensor(expected_result).view([n_samples, 1, self.k])
        input = np.repeat([self.cluster.labels[-1]], n_samples, axis=0).astype(np.float32)
        input = Tensor(input).view([n_samples, 1, self.N])
        result = self.model(input)
        self.assertTrue(result.equal(expected_result))

    def test_downsampling_with_channels_and_samples(self):
        n_chans = 3
        n_samples = 5
        result_aux = expected_result = np.unique([self.cluster.labels[-1]],axis=1).astype(np.float32)
        result_aux = np.repeat(expected_result, n_chans, axis=0)
        expected_result = zeros([n_samples, n_chans, self.k])
        input_aux = np.repeat([self.cluster.labels[-1]], n_chans, axis=0).astype(np.float32)
        input = zeros([n_samples, n_chans, self.N])
        for i in range(n_samples):
            expected_result[i,:,:] = Tensor(result_aux)
            input[i,:,:] = Tensor(input_aux)
        result = self.model(input)
        self.assertTrue(result.equal(expected_result))

    def test_downsampling_more_channels(self):
        n_chans = 10
        n_samples = 5
        result_aux = expected_result = np.unique([self.cluster.labels[-1]],axis=1).astype(np.float32)
        result_aux = np.repeat(expected_result, n_chans, axis=0)
        expected_result = zeros([n_samples, n_chans, self.k])
        input_aux = np.repeat([self.cluster.labels[-1]], n_chans, axis=0).astype(np.float32)
        input = zeros([n_samples, n_chans, self.N])
        for i in range(n_samples):
            expected_result[i,:,:] = Tensor(result_aux)
            input[i,:,:] = Tensor(input_aux)
        result = self.model(input)
        self.assertTrue(result.equal(expected_result))


class GraphUpsamplingTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        G_params = {}
        G_params['type'] = ds.SBM
        G_params['N']  = 256
        G_params['k']  = 4
        G_params['p'] = 0.15
        G_params['q'] = 0.01/4
        G_params['type_z'] = ds.RAND
        G = ds.create_graph(G_params, seed=SEED)
        nodes_dec = [4,16,32,64,256]
        ups = gc.WEI
        self.N = nodes_dec[-1]
        self.k = nodes_dec[0]
        self.cluster = gc.MultiResGraphClustering(G, nodes_dec, k=4, up_method=ups)
        self.model = Sequential()
        for i,U in enumerate(self.cluster.Us):
            self.add_layer(GraphUpsampling(U, self.cluster.As[i+1],gamma=1))

    def add_layer(self, module):
        self.model.add_module(str(len(self.model) + 1), module)

    def test_1D_upsampling(self):
        expected_result = self.cluster.labels[0].astype(np.float32)
        expected_result = Tensor(expected_result).view([1, 1, self.N])
        input = np.unique(self.cluster.labels[0]).astype(np.float32)
        input = Tensor(input).view([1, 1, self.k])
        result = self.model(input)
        self.assertTrue(result.equal(expected_result))

    def test_upsampling_with_channels(self):
        n_chans = 3
        expected_result = self.cluster.labels[0].astype(np.float32)
        expected_result = np.repeat([expected_result], n_chans, axis=0)
        expected_result = Tensor(expected_result).view([1, n_chans, self.N])
        input = np.unique(self.cluster.labels[0]).astype(np.float32)
        input = np.repeat([input], n_chans, axis=0)
        input = Tensor(input).view([1, n_chans, self.k])
        result = self.model(input)
        self.assertTrue(result.equal(expected_result))
    
    def test_upsampling_with_samples(self):
        n_samples = 5
        expected_result = self.cluster.labels[0].astype(np.float32)
        expected_result = np.repeat([expected_result], n_samples, axis=0)
        expected_result = Tensor(expected_result).view([n_samples, 1, self.N])
        input = np.unique(self.cluster.labels[0]).astype(np.float32)
        input = np.repeat([input], n_samples, axis=0)
        input = Tensor(input).view([n_samples, 1, self.k])
        result = self.model(input)
        self.assertTrue(result.equal(expected_result))

    def test_upsampling_with_channels_and_samples(self):
        n_chans = 3
        n_samples = 5
        result_aux = self.cluster.labels[0].astype(np.float32)
        result_aux = np.repeat([result_aux], n_chans, axis=0)
        expected_result = zeros([n_samples, n_chans, self.N])
        input_aux = np.unique(self.cluster.labels[0]).astype(np.float32)
        input_aux = np.repeat([input_aux], n_chans, axis=0)
        input = zeros([n_samples, n_chans, self.k])
        for i in range(n_samples):
            expected_result[i,:,:] = Tensor(result_aux)
            input[i,:,:] = Tensor(input_aux)
        result = self.model(input)
        self.assertTrue(result.equal(expected_result))

    def test_upsampling_more_channels(self):
        n_chans = 10
        n_samples = 5
        result_aux = self.cluster.labels[0].astype(np.float32)
        result_aux = np.repeat([result_aux], n_chans, axis=0)
        expected_result = zeros([n_samples, n_chans, self.N])
        input_aux = np.unique(self.cluster.labels[0]).astype(np.float32)
        input_aux = np.repeat([input_aux], n_chans, axis=0)
        input = zeros([n_samples, n_chans, self.k])
        for i in range(n_samples):
            expected_result[i,:,:] = Tensor(result_aux)
            input[i,:,:] = Tensor(input_aux)
        result = self.model(input)
        self.assertTrue(result.equal(expected_result))

if __name__ == "__main__":
    unittest.main()