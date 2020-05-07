import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import math

class trafficNetwork_Chengdu:
    def __init__(slef,):
        pass

    def network(edgelist):
        """
        generate street network
        """
        Chengdu = nx.MultiDiGraph()
        if velocity is None:
            for l, e in edgelist.iterrows():
                Chengdu.add_edge(int(e['Node_Start']),int(e['Node_End']),ID = int(e['Link']), length = float(e['Length']))
                Chengdu.nodes[int(e['Node_Start'])]['pos'] = (float(e['Longitude_Start']), float(e['Latitude_Start']))
        else:
            count=0
            for l, e in edgelist.iterrows():
                Chengdu.add_edge(int(e['Node_Start']),int(e['Node_End']),ID = int(e['Link']), length = float(e['Length']), velocity = float(velocity[count]))
                Chengdu.nodes[int(e['Node_Start'])]['pos'] = (float(e['Longitude_Start']), float(e['Latitude_Start']))
                count+=1
        return Chengdu

    def relative_velocity():
        """
        generate relative velocity
        """
        pass

    def threshold_q(Network, q):
        """
        remove edge which velocity lower than q
        """
        edge_properties = list(Network.edges.data('velocity'))
        for e in edge_properties:
            if e[2] < q: Network.remove_edge(*e[0:2])
        return Network

    def weakly_connected_cluster(Network):
        """
        get distribution of weakly connected cluster's size
        """
        return [len(c) for c in sorted(nx.weakly_connected_components(Network), key = len, reverse=True)]

    def histogram(dist):
        """
        generate hist array(linear full binning)
        """
        network_size = 1902 # Chengdu network
        hist = np.zeros(network_size)
        x_hist = np.arange(network_size)
        for i in dist: hist[i-1]+=1
        return x_hist, hist

    def histogram_logbinning(dist):
        """
        generate log binning hist array
        """
        maximum = int(math.log(dist[0], base))+1
        hist = np.zeros(maximum)
        x_hist = np.zeros(maximum)
        for x in dist: hist[int(math.log(x,base))]+=1
        for i in range(maximum): x_hist[i],hist[i] = (base**(i+1)+base**(i))*0.5,hist[i]/(base**(i+1)-base**i)
        return x_hist, hist
    ## linear regression
    def qr_householder(A):
        m, n = A.shape
        Q = np.eye(m) # Orthogonal transform so far
        R = A.copy() # Transformed matrix so far

        for j in range(n):
            # Find H = I - beta*u*u' to put zeros below R[j,j]
            x = R[j:, j]
            normx = np.linalg.norm(x)
            rho = -np.sign(x[0])
            u1 = x[0] - rho * normx
            u = x / u1
            u[0] = 1
            beta = -rho * u1 / normx

            R[j:, :] = R[j:, :] - beta * np.outer(u, u).dot(R[j:, :])
            Q[:, j:] = Q[:, j:] - beta * Q[:, j:].dot(np.outer(u, u))
        return Q, R
    def linearReg(A):
        m, n = A.shape
        AA = np.array([A[:,0], np.ones(m)]).T
        b = A[:, 1]

        Q, R = qr_householder(AA)
        b_hat = Q.T.dot(b)

        R_upper = R[:n, :]
        b_upper = b_hat[:n]

        x = np.linalg.solve(R_upper, b_upper)
        slope, intercept = x
        return slope, intercept

    def plot_slope():
    """
    generate plot and return slope
    """
        return a
