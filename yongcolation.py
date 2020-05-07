import numpy as np
import networkx as nx
import math
# 2-d lattice Network 생성
def Gen2dLatticeNetwork(size):
    Network = nx.grid_2d_graph(size,size)
    Node = np.array(Network.nodes)
    pos={tuple(Node[i]):[Node[i,0],Node[i,1]]for i in range(len(Node))}
    return Network
# rewiring
def rewiring_yong(Network,Probf):
    Network_edge = list(Network.edges)
    Network_node = list(Network.nodes)
    size = len(Network)
    for i in range(size):
        RandNum = np.random.rand(2)
        # change left node
        if Probf/2 <= RandNum[0] and RandNum[0] < Probf:
            Network.remove_edge(Network_edge[i][0],Network_edge[i][1])
            Network.add_edge(list(Network.nodes)[int(RandNum[1]*size)],Network_edge[i][1])
        # change right node
        elif 0<= RandNum[0] and RandNum[0] <Probf/2:
            Network.remove_edge(Network_edge[i][0],Network_edge[i][1])
            Network.add_edge(Network_edge[i][0],list(Network.nodes)[int(RandNum[1]*size)])
    return Network

# Netwrok X double swap 이용
def rewiringNX(Network,Probf):
    if len(Network.edges) !=0: nx.double_edge_swap(Network, Probf*len(Network.edges)*0.5, max_tries=len(Network.edges))
    else: pass
    return Network

# remove bond upper than Prob
def removeLinkUpperProb_2d(Network, Prob):
    Edge = list(Network.edges)
    for edge in Edge:
        if np.random.rand() >= Prob: Network.remove_edge(edge[0],edge[1])
    return Network

def removeNodeUpperProb_2d(Network, Prob):
    Node = list(Network.nodes)
    for node in Node:
        if np.random.rand() >= Prob: Network.remove_node(node)
    return Network

# get cc Size
def ccSize(Network):
    return [len(c) for c in sorted(nx.connected_components(Network), key=len, reverse=True)]

# ensemble experiment
def getClusterSize(LatticeSize,probf,timeEnsemble,Step):
    occupiedProb = np.linspace(0,1,Step+1)
    CC = np.zeros([2,Step+1])      # [0]:GCC, [1]:SCC
    c=0
    for prob in occupiedProb:
        for i in range(timeEnsemble):
            ClusterSize = ccSize(rewiringNX(removeLinkUpperProb_2d(Gen2dLatticeNetwork(LatticeSize),prob),probf))
            if len(ClusterSize)==1:CC[0,c] += ClusterSize[0]/timeEnsemble
            elif len(ClusterSize)>=2:
                CC[0,c] += ClusterSize[0]/timeEnsemble
                CC[1,c] += ClusterSize[1]/timeEnsemble
        c+=1
    return CC

# linear logbinning
def linearBinning(dist,max_hist):
    # maximum = dist[0]
    # hist
    hist = np.zeros(max_hist)
    for i in dist:
        hist[i-1]+=1
    # x-axis of hist
    x_hist = np.arange(max_hist)+1

    # hist_re=hist[0][np.where(hist[0]!=0)]
    # x_hist=x_hist[np.where(hist[0]!=0)]
    return x_hist, hist

# log binning
def logBinning(dist,base):
    # histogram
    maximum=int(math.log(dist[0],base))+1
    hist=np.zeros(maximum)
    # add cluster size each range
    for x in dist:
        hist[int(math.log(x,base))]+=1
    # generate x axis
    x_hist=np.zeros(maximum)
    for i in range(maximum):
        x_hist[i]=(base**(i+1)+base**(i))*0.5
    # divide by range
    for i in range(maximum):
        hist[i]/=(base**(i+1)-base**i)
    return x_hist,hist

# linear Regression (use linearReg)
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
