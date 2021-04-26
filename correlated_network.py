import numpy as np
import networkx as nx
import random
from scipy import stats

def cluster_distribution(network):
    return [len(c) for c in sorted(nx.connected_components(network), key=len, reverse=True)]

class correlated_network:
    def __init__(self, size):
        self.size = size
        
        # lattice
        self.G = nx.grid_2d_graph(self.size,self.size, periodic=True)
        Node = np.array(self.G.nodes)
        pos={tuple(Node[i]):[Node[i,0],Node[i,1]]for i in range(len(Node))}
        # uniform weight
        for u,v in list (self.G.edges):
            self.G[u][v]['weight'] = random.random()
        
        self.edge = list(self.G.edges(data='weight'))
        
    # 2-d lattice Network 생성
    def reconstruct_lattice_by_original_weight(self):
        self.G = nx.grid_2d_graph(self.size,self.size)
        Node = np.array(self.G.nodes)
        pos={tuple(Node[i]):[Node[i,0],Node[i,1]]for i in range(len(Node))}
        for e in self.edge:
            self.G.edges[e[0], e[1]]['weight']=e[2]
    
    # check neighbor link
    def neighbor_link(self, link):
        """
        link : (u, v, weight)
        
        """
        neighbor = list(self.G.edges(link[0], data = 'weight'))\
                    + list(self.G.edges(link[1], data = 'weight'))
        # 자기 자신 두번 불러지므로 제거
        neighbor.remove(link)
        neighbor.remove((link[1], link[0], link[2]))
        return neighbor
    
    def get_neighbor_weight_mean(self, link):
        neighbor_mean = 0
        n_link = self.neighbor_link(link)
        for i in n_link:
            neighbor_mean+=i[2]
        return link[2], neighbor_mean/len(n_link)
    
    def link_weight_distance(self, edge_ij, edge_mn, shuffle=False):
        if shuffle:
            w_mn, w_ij_neighbor = self.get_neighbor_weight_mean(edge_ij)
            w_ij, w_mn_neighbor = self.get_neighbor_weight_mean(edge_mn)
        else:
            w_ij, w_ij_neighbor = self.get_neighbor_weight_mean(edge_ij)
            w_mn, w_mn_neighbor = self.get_neighbor_weight_mean(edge_mn)
        return ((w_ij - w_ij_neighbor)**2 + (w_mn - w_mn_neighbor)**2)*0.5
        
    def shuffling_weight_to_correlated(self):
        """
        weight 셔플
    
        """
        link_ij, link_mn = self.edge[random.randint(0,self.G.number_of_edges()-1)], \
                        self.edge[random.randint(0,self.G.number_of_edges()-1)]
        original_state = self.link_weight_distance(link_ij, link_mn)
        shuffle_state = self.link_weight_distance(link_ij, link_mn,shuffle=True)
        if original_state > shuffle_state:
            self.G.edges[link_ij[0], link_ij[1]]['weight'], self.G.edges[link_mn[0], link_mn[1]]['weight']=\
            self.G.edges[link_mn[0], link_mn[1]]['weight'], self.G.edges[link_ij[0], link_ij[1]]['weight']
            self.edge = list(self.G.edges(data='weight'))
    
    def get_nn_pearson_correlation(self):
        """
        pearson correlation 확인
        
        """
        weight_ij = []
        weight_neighbor = []
        for e in self.edge:
            w, w_neigh = self.get_neighbor_weight_mean(e)
            weight_ij.append(w)
            weight_neighbor.append(w_neigh)
        return stats.pearsonr(weight_ij, weight_neighbor)
    