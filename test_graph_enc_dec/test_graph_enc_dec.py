import unittest
import sys
import numpy as np

sys.path.insert(0, '.')
from graph_enc_dec import data_sets as ds
from graph_enc_dec import graph_clustering as gc

SEED = 15

# TODO: test graph creation

class DiffusedSparse2GSTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        G_params = {}
        G_params['type'] = ds.SBM
        G_params['N']  = 32
        G_params['k']  = 4
        G_params['p'] = 0.7
        G_params['q'] = 0.1
        G_params['type_z'] = ds.RAND
        eps1 = 0.1
        eps2 = 0.3
        self.Gx, self.Gy = ds.perturbated_graphs(G_params, eps1, eps2, seed=SEED)

    def test_to_unit_norm(self):
        n_samps = [50, 10, 10]
        L = 6
        n_delts = 4
        data = ds.DiffusedSparse2GS(self.Gx, self.Gy, n_samps, L, n_delts)
        data.to_unit_norm()

        for i in range(n_samps[0]):
            self.assertAlmostEqual(np.linalg.norm(data.train_X[:,i]),1)
            self.assertAlmostEqual(np.linalg.norm(data.train_Y[:,i]),1)
        for i in range(n_samps[1]):
            self.assertAlmostEqual(np.linalg.norm(data.val_X[:,i]),1)
            self.assertAlmostEqual(np.linalg.norm(data.val_Y[:,i]),1)
        for i in range(n_samps[2]):
            self.assertAlmostEqual(np.linalg.norm(data.test_X[:,i]),1)
            self.assertAlmostEqual(np.linalg.norm(data.train_Y[:,i]),1)
    
# Check that repeated values doesnt affect the obtained clust size
class GraphClusteringTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(SEED)
        self.G_params = {}
        self.G_params['type'] = ds.SBM
        self.G_params['N']  = 256
        self.G_params['k']  = 4
        self.G_params['p'] = 0.15
        self.G_params['q'] = 0.01/4

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



if __name__ == "__main__":
    unittest.main()