import numpy as np
import networkx as nx
import os
import re
import matplotlib.pyplot as plt
import math

# edgelist data convert to npy
def convert2npy_edgelist(path,filename):
    """Short summary.
    Chengdu road linklist's raw data is csv format. This function convert to npy format.

    Parameters
    ----------
    path : string ex) '/home/dataset/'
        path of raw csv data

    filename : string ex) 'ChengduLink'
        new name of new file


    Returns
    -------
    type filename.npy
        np.array(dtype=[('Link', 'int'), ('Node_Start', 'int'), ('Longitude_Start', 'float'),
            ('Latitude_Start', 'float'),('Node_End', 'int'), ('Longitude_End', 'float'),('Latitude_End', 'float'),('LENGTH', 'float')])

    """
    data = np.genfromtxt(path, delimiter=',',skip_header=1,dtype=[('Link', 'int'), ('Node_Start', 'int'), ('Longitude_Start', 'float'),('Latitude_Start', 'float'),('Node_End', 'int'), ('Longitude_End', 'float'),('Latitude_End', 'float'),('LENGTH', 'float')])
    np.save(filename,data)

# speed data convert to npy
def convert2npy_linkspeed(Path):
    """Short summary.
    Chengdu road speed data's raw data is csv format. This function convert to npy format.

    Parameters
    ----------
    Path : string ex) '/home/dataset'
        path of raw csv data

    Returns
    -------
    type speed[monthDay]_[1or0].npy
        np.array(dtype=[('Period','U12'),('Link','int'),('Speed','float')])

    """
    path_dr = Path
    file_list = os.listdir(path_dr)
    file_list.sort()
    file_list = file_list[5:len((file_list))]
    for i in range(len(file_list)):
        path=file_list[i]
        data = np.genfromtxt(os.path.join(Path,path),delimiter=',',skip_header=1, dtype=[('Period','U12'),('Link','int'),('Speed','float')])
        filename=int(re.findall('\d+',path)[0])
        fileType=int(re.findall('\d+',path)[1])
        np.save('speed[{}]_[{}].npy'.format(filename,fileType),data)

# generate Street network
def genStreetNet(Edgelist):
    """Short summary.
    Geneate road network with 'No' direct.
    Parameters
    ----------
    Edgelist : np.array(dtype=[('Link', 'int'), ('Node_Start', 'int'), ('Longitude_Start', 'float'),
        ('Latitude_Start', 'float'),('Node_End', 'int'), ('Longitude_End', 'float'),('Latitude_End', 'float'),('LENGTH', 'float')])


    Returns
    -------
    type Graph()


    """
    # node label & number
    node_list = np.unique(Edgelist['Node_Start'])
    # network generating
    G = nx.DiGraph()
    # add nodes
    G.add_nodes_from(node_list)
    # add edges
    for i in range(len(Edgelist)):
        G.add_edge(Edgelist['Node_Start'][i],Edgelist['Node_End'][i],label=Edgelist['Link'][i])
    return G

# generate newowrk nodes' position
def network_pos(Edgelist):
    """Short summary.
    Generate network node position.
    ex) pos=network_pos(Edgelist)
        nx.draw_networkx(network, pos=pos, ...)

    Parameters
    ----------
    Edgelist : np.array(dtype=[('Link', 'int'), ('Node_Start', 'int'), ('Longitude_Start', 'float'),
        ('Latitude_Start', 'float'),('Node_End', 'int'), ('Longitude_End', 'float'),('Latitude_End', 'float'),('LENGTH', 'float')])

    Returns
    -------
    type tuple


    """
    # assign pos for nodes
    return {i:[Edgelist[Edgelist['Node_Start']==i]['Longitude_Start'][0],Edgelist[Edgelist['Node_Start']==i]['Latitude_Start'][0]]for i in range(len(np.unique(Edgelist['Node_Start'])))}

# get max velocity each street link
def Max_velocity(velocity0,velocity1):
    """Short summary.
    Find each link's fatest speed in whole day.
    The reason why it takes two speed array is Chengdu's speed data splice day in two Period. 03:00~13:00, 13:00~23:00
    Parameters
    ----------
    velocity0 : np.array(dtype=[('Period','U12'),('Link','int'),('Speed','float')])
    velocity1 : np.array(dtype=[('Period','U12'),('Link','int'),('Speed','float')])

    Returns
    -------
    type np.array(dtype=[('Period','U12'),('Link','int'),('Speed','float')])

    """
    max_velo = np.zeros(len(np.unique(velocity0['Link'])))
    for i in range(len(max_velo)):
        max_velo[i] = max([max(velocity0[velocity0['Link'] == i+1]['Speed']),max(velocity1[velocity1['Link'] == i+1]['Speed'])])
    return max_velo

# get relative velocity
def relativeVelocity(Period,velocity0,velocity1):
    """Short summary.
    Divide road's each period speed by Fastest speed, get relative velocity each road

    Parameters
    ----------
    Period : string
        ex) '03:00-03:02'
    velocity0 : np.array(dtype=[('Period','U12'),('Link','int'),('Speed','float')])
    velocity1 : np.array(dtype=[('Period','U12'),('Link','int'),('Speed','float')])

    Returns
    -------
    type np.array(dtype=[('Period','U12'),('Link','int'),('Speed','float')])


    """
    return np.array(velocity0[velocity0['Period']==Period]['Speed']/Max_velocity(velocity0,velocity1))

# generate network given weight by relative speed
def genStreetNet_speed(Edgelist,reVelo):
    """Short summary.
    Generate road network assigned relative velocity as weight on each link

    Parameters
    ----------
    Edgelist : np.array(dtype=[('Link', 'int'), ('Node_Start', 'int'), ('Longitude_Start', 'float'),
        ('Latitude_Start', 'float'),('Node_End', 'int'), ('Longitude_End', 'float'),('Latitude_End', 'float'),('LENGTH', 'float')])

    reVelo : type np.array(dtype=[('Period','U12'),('Link','int'),('Speed','float')])

    Returns
    -------
    type Graph()

    """
    # node label & number
    node_list = np.unique(Edgelist['Node_Start'])
    # network generating
    G = nx.DiGraph()
    # add nodes
    G.add_nodes_from(node_list)
    # add edges
    for i in range(len(Edgelist)):
        G.add_edge(Edgelist['Node_Start'][i],Edgelist['Node_End'][i],label=Edgelist['Link'][i],weight=reVelo[i])
    return G

# remove link under parameter q
def remove_qRoad(q,Edgelist,reVelo):
    """Short summary.
    Generate road network that cutted links(roads) which weight(relative velocity) smaller than q

    Parameters
    ----------
    q : float

    Edgelist : np.array(dtype=[('Link', 'int'), ('Node_Start', 'int'), ('Longitude_Start', 'float'),
        ('Latitude_Start', 'float'),('Node_End', 'int'), ('Longitude_End', 'float'),('Latitude_End', 'float'),('LENGTH', 'float')])

    reVelo : type np.array(dtype=[('Period','U12'),('Link','int'),('Speed','float')])

    Returns
    -------
    type Graph()

    """
    orign_net = genStreetNet_speed(Edgelist,reVelo)
    return_net = genStreetNet_speed(Edgelist,reVelo)
    Edge = np.array(orign_net.edges)
    for i in range(len(Edge)):
        if list(orign_net.edges.data('weight'))[i][2] < q:
            return_net.remove_edge(*Edge[i])
    return return_net

# get weakly connected components
def weaklycc(network):
    """Short summary.
    Generate weakly connected cluster distribution

    Parameters
    ----------
    network : Graph()

    Returns
    -------
    type list

    """
    return [len(c) for c in sorted(nx.weakly_connected_components(network), key=len, reverse=True)]

# measuring GCC, SCC, CPoint, and generating graph
def criticalGraph(day,Period,edgelist,speedlist0,speedlist1):
    """Short summary.
    calculate critical q point when second giant connected component was max.
    Parameters
    ----------
    day : string
        ex) '10' input data's day
    Period : string
        ex) '08:00-08:02' time period
    edgelist : np.array(dtype=[('Link', 'int'), ('Node_Start', 'int'), ('Longitude_Start', 'float'),
        ('Latitude_Start', 'float'),('Node_End', 'int'), ('Longitude_End', 'float'),('Latitude_End', 'float'),('LENGTH', 'float')])
    speedlist0 : np.array(dtype=[('Period','U12'),('Link','int'),('Speed','float')])
    speedlist1 : np.array(dtype=[('Period','U12'),('Link','int'),('Speed','float')])

    Returns
    -------
    type
        fgure.png
    """
    # relative velocity
    rv = relativeVelocity(Period,speedlist0,speedlist1)
    # get GCC, SCC each q
    h = 101
    cc = np.zeros([2,h])
    # control parameter q
    q,a = np.linspace(0,1,h),0
    for i in q:
        dist=weaklycc(remove_qRoad(i,edgelist,rv))
        cc[0,a] = dist[0]
        if len(dist)>=2:cc[1,a]=dist[1]
        a+=1
    # get critical point(SCC max size)
    criticalPoint=q[np.where(cc[1]==max(cc[1]))][0]
    # Graph
    fig, ax1 = plt.subplots(figsize=(12,9))
    ax2 = ax1.twinx()
    curve1 = ax1.errorbar(q,cc[0]/1902,marker='s',markersize=20,label='GCC')
    curve2 = ax2.errorbar(q,cc[1]/1902,marker='^',markersize=20,label='SCC',c='orange')
    curves=[curve1,curve2]
    ax1.legend(curves,[curve.get_label()for curve in curves],fontsize='x-large')
    plt.savefig('Chengdu_june{}_{}_ciritcalpoint_{}.png'.format(day,Period,criticalPoint),transparent=True,dpi=300)
    plt.close()

# log binning
def logBinning(dist,base):
    """Short summary.
    Generate logbinning histogram array tuple.

    Parameters
    ----------
    dist : list
        ex) clsuter size distribution
    base : int
        ex) log_{base}
    Returns
    -------
    type
        Description of returned object.

    """
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

# tau measuring
def dist2hist_linear(dist,dropSize=0):
    hist=linearBinning(dist,1902)
    hist_re=hist[1][np.where(hist[1]>dropSize)]
    x_hist=hist[0][np.where(hist[1]>dropSize)]
    linear_hist=[list(x_hist),list(hist_re)]
    a,b=linearReg(np.log10(np.array(linear_hist).T[1:]))
    plt.scatter(hist[0],hist[1])
    plt.plot([10**(-1),10**(3)],[10**((-1)*a+b),10**(3*a+b)],'r:')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-1,1e4)
    plt.ylim(1e-1,1e4)
    plt.show()
    print('tau : ',-a)

def rv_hourRange_generator(data_speed):
    dayVelocity=[]
    for i in range(10):
        time = np.unique(data_speed['Period'])[30*i:30*(i+1)]
        velocity = data_speed[np.where(data_speed['Period']==time[0])]
        velocity['Period'] = time[0][0:2]
        velocity['Speed']/=30
        for period in time[1:]: velocity['Speed'] += data_speed[np.where(data_speed['Period']==period)]['Speed']/30
        dayVelocity.append(velocity)
    dayVelocity = np.vstack(dayVelocity)

    max_velo = np.zeros(len(np.unique(data_speed['Link'])))
    for i in range(len(max_velo)):
        max_velo[i] = max(data_speed[data_speed['Link'] == i+1]['Speed'])

    for i in range(len(dayVelocity)):
        dayVelocity[i]['Speed']/=max_velo
    return dayVelocity
