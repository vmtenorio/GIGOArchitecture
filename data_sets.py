import numpy as np
import pandas as pd
import torch
from cnngs_src import graphtools, datatools

HOURS_A_DAY = 24

class SourceLocalization:
    """
    Class for source localization problem.
    Generates SBM graph and data and labels for the NN
    Constructor args:
        N_nodes: number of nodes of the graph
        N_comm: number of communities
        p_ii: probability for nodes in the same community
        p_ij: probability for edges between nodes from different communities
    """
    def __init__(self, N_nodes, N_comm, p_ii, p_ij, N_samples, maxdiff):
        self.N_nodes = N_nodes
        self.N_comm = N_comm
        self.N_samples = N_samples

        # Create the GSO
        self.mapping = graphtools.create_mapping()
        self.graph = graphtools.create_SBM(N_nodes, p_ii, p_ij)

        # Generate the data
        self.train_labels = np.ceil(self.N_comm*np.random.rand(self.N_samples))
        self.test_labels = np.ceil(N_classes*np.random.rand(N_test))
        self.train_data = datatools.create_samples(self.train_labels, self.graph, maxdiff)
        self.test_data = datatools.create_samples(self.test_labels, self.graph, maxdiff)

    def create_SBM(self, N_nodes, p_ii, p_ij):
        #TODO
        raise NotImplementedError

    def create_mapping(self):
        #TODO
        raise NotImplementedError

    def indices_to_onehot(a):
    	n_samples = a.size # Number of samples
    	E = np.zeros((self.N_nodes,n_samples)) # Deltas to diffuse
    	for it in range(n_samples):
    		E[(a[it]-1).astype(int),it] = 1.
    	return E

    def create_samples(a,K):
    	n_samples = a.size # Number of samples
    	E = self.indices_to_onehot(a)
    	t = np.floor(K*np.random.rand(n_samples)) # Diffusion times
    	# Storage:
    	x = np.zeros((self.N_nodes,n_samples))
    	#!!!
    	for it in range(n_samples):
    		St = np.linalg.matrix_power(self.S,t[it].astype(int))
    		x[:,it] = St.dot(E[:,it])
    	return x

class GIGOMaxComm:
    """
    Class that defines synthetic data for the GIGO problem.
    The data is defined over an SBM graph and the labels (in the communities
    graph) are the maximum value in that community
    """
    def __init__(self, N_nodes, N_comm, p_ii, p_ij, N_samples, train_test_coef, limit_data):
        self.N_nodes = N_nodes
        self.N_comm = N_comm
        self.N_samples = N_samples

        # Create the GSO
        self.mapping, self.nodes_com = graphtools.create_mapping_NC(self.N_nodes, self.N_comm)
        self.Ngraph_unnorm, self.Cgraph_unnorm = graphtools.create_SBMc(self.N_nodes, self.N_comm, p_ii, p_ij, self.mapping)
        self.Ngraph = graphtools.norm_graph(self.Ngraph_unnorm)
        self.Cgraph = graphtools.norm_graph(self.Cgraph_unnorm)

        # Generate the data. Defined over nodes graph
        self.data = np.ceil(limit_data*np.random.rand(self.N_samples, self.N_nodes))
        self.data = graphtools.lp_filter(self.data, self.Ngraph_unnorm)
        # Generate labels, defined over communities graph
        self.labels = datatools.generate_labels_GIGOMax(self.data, self.mapping, self.nodes_com)

        self.train_data, self.train_labels, self.test_data, self.test_labels = \
            datatools.train_test_split(self.data, self.labels, train_test_coef)

        self.turn_to_tensors()

    def turn_to_tensors(self):
        # Turn data into tensors
        self.train_data = torch.FloatTensor(self.train_data)
        self.train_labels = torch.FloatTensor(self.train_labels)
        self.test_data = torch.FloatTensor(self.test_data)
        self.test_labels = torch.FloatTensor(self.test_labels)

class FlightData:
    def __init__(self, graph_route, data_route, train_test_coef, fut_time_pred):
        self.S = np.loadtxt(graph_route, delimiter=',')
        self.N_nodes = self.S.shape[0]
        self.train_test_coef = train_test_coef
        self.fut_time_pred = fut_time_pred

        self.read_data(data_route, train_test_coef)

    def read_data(self, data_route, train_test_coef):
        df = pd.read_csv(data_route)

        self.airports = df.columns[5:] # Delete year month day and hour
        assert len(self.airports) == self.N_nodes

        self.ndata = len(df.index)

        self.ndays = int(self.ndata / HOURS_A_DAY)
        self.n_samples = self.ndays * (HOURS_A_DAY - 2)
        self.data = np.zeros([self.n_samples, self.N_nodes, 2])
        self.labels = np.zeros([self.n_samples, self.N_nodes])
        c = 0
        for d in range(self.ndays):
            for h in range(24-self.fut_time_pred):
                it = HOURS_A_DAY * d + h
                self.data[c,:,0] = list(df.loc[it])[5:]
                self.data[c,:,1] = [df.loc[it,'Day'].item()] * self.N_nodes
                self.labels[c,:] = list(df.loc[it+self.fut_time_pred])[5:]
                c += 1
        assert c == self.n_samples

        #self.convert_labels()
        self.train_data, self.train_labels, self.test_data, self.test_labels = \
            datatools.train_test_split(self.data, self.labels, train_test_coef)
        self.turn_to_tensors()

    def convert_labels(self):
        """
        This proc converts the labels from numbers to binary.
        There is delay or not, 0 or 1
        """
        self.labels = self.labels > 0
        self.labels = self.labels.astype(int)

    def turn_to_tensors(self):
        # Turn data into tensors
        self.train_data = torch.FloatTensor(self.train_data)
        self.train_labels = torch.FloatTensor(self.train_labels)
        self.test_data = torch.FloatTensor(self.test_data)
        self.test_labels = torch.FloatTensor(self.test_labels)
