from torch import manual_seed
from graph_clustering import NONE, NO_A, BIN, WEI

class GraphEncoderDecoder():
    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)

    def __init__(self,
                features_d,
                features_u,
                nodes_d,
                nodes_u,
                ascendance,
                descendance,
                hierA_d,
                hierA_u,
                upsampling,
                downsampling):
        print("INIT NETWORK")
        if features_d[-1] != features_u[0] or nodes_d[-1] != nodes_u[0]:
            raise RuntimeError('Different definition of dimension for the latent space')

        self.fts_down = features_d
        self.fts_up = features_u
        self.nodes_d = nodes_d
        self.nodes_u = nodes_u
        self.asc = ascendance
        self.desc = descendance
        self.hierA_d = hierA_d
        self.hierA_u = hierA_u
        self.kernel_size = 1
        self.ups = upsampling
        self.downs = downsampling
        # Input?

    def build_network(self):
        # Downsampling Section
        # Input always must have only 1 feature
        self.add_layer(nn.Conv1d(1, self.fts_down[0]
                        self.kernel_size, bias=False))
        self.add_layer(nn.Tanh())

        for l in range(len(self.fts_down)-1):
            self.add_layer(nn.Conv1d(self.fts_down[l], self.fts_down[l+1], 
                        self.kernel_size, bias=False))
            # Maybe change downs==None for number of nodes of layer l and l+1
            # Always forcing one layer without downsAMPLING
            if l > 1 and self.ups != None:
                #A = None if self.downs == 'no_A' or self.downs == 'original' else self.hier_A[l+1]
                self.add_layer(GraphDownsampling(self.asc[l-1], self.nodes_d[l-1]))

            self.add_layer(nn.BatchNorm1d(self.fts_down[l+1]))
            self.add_layer(nn.Tanh())

        # Upsampling Section
        for l in range(len(sel.fts_up)-1):
            self.add_layer(nn.Conv1d(self.fts_up[l], self.fts_up[l+1], 
                        self.kernel_size, bias=False))

            # Maybe change ups==None for number of nodes of layer l and l+1
            # Always forcing one layer without upsampling
            if l < len(sel.fts_up)-2 and self.ups != NONE:
                A = None if self.ups == NO_A or self.ups == REG else self.hier_A[l+1]
                # Add layer graph upsampling
                self.add_layer(GraphUpsampling(self.desc[l], self.nodes_u[l],
                                               A, self.upsampling, self.gamma))
            # Intermediate act function
            self.add_layer(nn.Tanh())
            self.add_layer(nn.BatchNorm1d(self.fts_up[l+1]))
        
        # Output always must have only 1 feature
        self.add_layer(nn.Conv1d(self.fts_up[-1], 1, 
                        self.kernel_size, bias=False))

        # Last activation function
        self.add_layer(nn.Tanh())


"""
Use information from the agglomerative hierarchical clustering for doing the upsampling by
creating the upsampling matrix U
"""
class GraphUpsampling(nn.Module):
    def __init__(self, descendance, parent_size, A, method, gamma):
        super(GraphUpsampling, self).__init__()
        self.descendance = descendance
        self.parent_size = parent_size
        self.A = A
        if A is not None:
            D_inv = np.linalg.inv(np.diag(np.sum(self.A,0)))
            self.A_mean = D_inv.dot(self.A)
        self.method = method
        self.gamma = gamma
        self.child_size = len(descendance)
        self.create_U()

    def create_U(self):
        self.U = np.zeros((self.child_size, self.parent_size))
        for i in range(self.child_size):
            self.U[i,self.descendance[i]-1] = 1
        self.U = torch.Tensor(self.U)#.to_sparse()

    def forward(self, input):
        # TODO: check if making ops with np instead of torch increase speed
        n_channels = input.shape[1]
        matrix_in = input.view(self.parent_size, n_channels)

        parents_val = self.U.mm(matrix_in)
        # NOTE: gamma = 1 is equivalent to no_A
        # NOTE: gamma = 0 is equivalent to the prev setup
        if self.method == 'no_A':
            output = parents_val
        elif self.method == 'binary':
            neigbours_val = torch.Tensor(self.A_mean).mm(parents_val)
#            neigbours_val = torch.Tensor(self.A/np.sum(self.A,0)).mm(parents_val)
            output = self.gamma*parents_val + (1-self.gamma)*neigbours_val
        elif self.method == 'weighted':
            neigbours_val = torch.Tensor(self.A_mean).mm(parents_val)
            output =  self.gamma*parents_val + (1-self.gamma)*neigbours_val
        elif self.method == 'original':
            sf = self.child_size/self.parent_size
            output = torch.nn.functional.interpolate(input, scale_factor=sf,
                                    mode='linear', align_corners=True)
        else:
            raise RuntimeError('Unknown sampling method')
        return output.view(1, n_channels, self.child_size)

class GraphDownsampling(nn.Module):
    def __init__(self, ascendance, child_size):
        super(GraphUpsampling, self).__init__()
        self.asc = ascendance
        self.parent_size = len(ascendance)
        self.child_size = child_size
        self.create_D()

    def create_D(self):
        self.D = np.zeros((self.child_size, self.parent_size))
        for i in range(self.child_size):
            indexes = np.where(self.asc == i+1)
            self.D[i,indexes] = 1
        Deg_inv = np.linalg.inv(np.diag(np.sum(self.D,1)))  
        self.D = torch.Tensor(Deg_inv.dot(self.D))

    def forward(self, input):
        #n_channels = input.shape[1]
        #matrix_in = input.view(self.parent_size, n_channels)
        #parents_val = self.U.mm(matrix_in)
        #return output.view(1, n_channels, self.child_size)
        return self.U.mm(input)
        