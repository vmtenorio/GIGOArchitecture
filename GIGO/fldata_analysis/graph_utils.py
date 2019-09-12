import numpy as np
import pandas as pd

def create_airport_graph(df, airports):

    n_airports = len(airports)
    #print(n_airports)

    adj_mat = np.zeros([n_airports, n_airports])

    df2 = df.copy()

    for a in range(n_airports):
        airport = airports[a]
        for a2 in range(a+1, n_airports):
            airport2 = airports[a2]

            adj_mat[a][a2] = len(df2[(df2['ORIGIN_AIRPORT_ID'] == airport) & (df2['DEST_AIRPORT_ID'] == airport2)].index)
            adj_mat[a2][a] = len(df2[(df2['ORIGIN_AIRPORT_ID'] == airport2) & (df2['DEST_AIRPORT_ID'] == airport)].index)

        df2.drop(df2[(df2['ORIGIN_AIRPORT_ID'] == airport) | (df2['DEST_AIRPORT_ID'] == airport)].index, inplace=True)

    #print(adj_mat)
    return adj_mat

def my_eig(S):
	d,V = np.linalg.eig(S)
	order = np.argsort(-d)
	d = d[order]
	V = V[:,order]
	D = np.diag(d)
	VV = np.linalg.inv(V)
	SS = V.dot(D.dot(VV))
	diff = np.absolute(S-SS)
	if diff.max() > 1e-6:
		print("Eigendecomposition not good enough")
	return V,D

def norm_graph(A):
    """Receives adjacency matrix and returns normalized (divided by biggest eigenvalue) """
    V,D = my_eig(A)
    if np.max(np.abs(np.imag(V))) < 1e-6:
        V = np.real(V)
    if np.max(np.abs(np.imag(D))) < 1e-6:
        D = np.real(D)
    d = np.diag(D)
    dmax = d[0]
    return (A/dmax).astype(np.float32)
