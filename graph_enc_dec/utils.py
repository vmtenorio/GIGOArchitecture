import numpy as np
import matplotlib.pyplot as plt
from pygsp.graphs import Graph, StochasticBlockModel, ErdosRenyi

# Graph Type Constants
SBM = 1
ER =  2

# Comm Node Assignment Constants
CONT = 1    # Contiguous nodes
ALT =  2    # Alternated nodes
RAND = 3    # Random nodes

"""
Distribute contiguous nodes in the same community while assuring that all
communities have (approximately) the same number of nodes.  
"""
def assign_nodes_to_comms(N,k):
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

def plot_graph_clusters(G, labels, n_clusts):
    _, axes = plt.subplots(1, n_clusts)
    G.set_coordinates(kind='community2D')
    for i in range(n_clusts):
        G.plot_signal(labels[i], ax=axes[i])
    plt.show()

def create_graph(ps, seed=None, type_z=RAND):
    if ps['type'] == SBM:
        if type_z == CONT:
            z = assign_nodes_to_comms(ps['N'],ps['k'])
        elif type_z == ALT:
            z = np.array(list(range(ps['k']))*int(ps['N']/ps['k'])+list(range(ps['N']%ps['k'])))
        elif type_z == RAND:
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