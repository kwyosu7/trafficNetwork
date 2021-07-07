import numpy as np
import networkx as nx
import random
from scipy import stats

def cluster_distribution(network):
    return [len(c) for c in sorted(nx.connected_components(network), key=len, reverse=True)]

class correlated_network:
    def __init__(self, size, wegiht_list = None, Periodic=True):
        self.size = size
        self.Periodic = Periodic
        # lattice
        self.lattice = nx.grid_2d_graph(self.size, self.size, periodic=self.Periodic)
        self.Nodes = list(self.lattice.nodes)
        self.pos={tuple(self.Nodes[i]):[self.Nodes[i][0],self.Nodes[i][1]]for i in range(len(self.Nodes))}
        self.Edges = list(self.lattice.edges())
        if weight_list = None:
            self.edge_list_weight = self.edge_list_weight_gen()
        else:
            self.edge_list_weight = weight_list
        self.edge_neighbor = self.edge_neighbor_gen()
    
    def edge_align(self, edge_list):
        """
        my link rule (small, large)
        
        input : link 'list'
        return : link 'list'
        
        """
        new_edge_list = []
        for e in edge_list:
            n1, n2 = e[0], e[1]
            if n1[0] > n2[0]: new_e = (e[1], e[0])
            elif n1[0] == n2[0]:
                if n1[1] > n2[1]:new_e = (e[1], e[0])
                else: new_e = e
            else: new_e = e
            new_edge_list.append(new_e)
        return new_edge_list
    
    def edge_list_weight_gen(self):
        """
        return edge, weight array (uniform distribution)
        """
        edge_list_weight, w_list, c = {}, np.random.uniform(0,1,len(self.Edges)), 0
        for e in self.Edges:
            u = e[0]
            v = e[1]
            edge_list_weight[(u,v)] = w_list[c] 
            c+=1
        return edge_list_weight
    
    def edge_neighbor_gen(self):
        """
        return edge neighbor list
        """
        edge_neighbor = {}
        for e in self.Edges:
            node_i_links = list(self.lattice.edges(e[0]))
            node_j_links = list(self.lattice.edges(e[1]))
            neighbors = self.edge_align(node_i_links + node_j_links)
            for i in range(2):neighbors.remove(e)
            edge_neighbor[e]=neighbors
        return edge_neighbor
    
    def case_shuffle_weight(self, e_ij, e_mn, shuffle = False):
        if shuffle:
            w_mn, w_ij_neigh = self.edge_list_weight[e_ij], self.neighbor_weight_mean(e_ij)
            w_ij, w_mn_neigh = self.edge_list_weight[e_mn], self.neighbor_weight_mean(e_mn)
        else:
            w_ij, w_ij_neigh = self.edge_list_weight[e_ij], self.neighbor_weight_mean(e_ij)
            w_mn, w_mn_neigh = self.edge_list_weight[e_mn], self.neighbor_weight_mean(e_mn)
        return ((w_ij - w_ij_neigh)**2 + (w_mn - w_mn_neigh)**2)*0.5
    
    def neighbor_weight_mean(self, edge):
        neighbor_list = self.edge_neighbor[edge]
        weight_list = []
        for e in neighbor_list:
            weight_list.append(self.edge_list_weight[e])
        return np.mean(np.array(weight_list))
    
    def shuffle_weight(self):
        link_ij, link_mn = self.Edges[random.randint(0,self.lattice.number_of_edges()-1)], \
                        self.Edges[random.randint(0,self.lattice.number_of_edges()-1)]
        original_state = self.case_shuffle_weight(link_ij, link_mn)
        shuffle_state = self.case_shuffle_weight(link_ij, link_mn, shuffle = True)
        if original_state > shuffle_state:
            self.edge_list_weight[link_ij], self.edge_list_weight[link_mn] = self.edge_list_weight[link_mn], self.edge_list_weight[link_ij]
    
    def get_nn_pearson_correlation(self):
        """
        pearson correlation 확인

        """
        weight_ij = []
        weight_neighbor = []
        for e in self.Edges:
            w, w_neigh = self.edge_list_weight[e], self.neighbor_weight_mean(e)
            weight_ij.append(w)
            weight_neighbor.append(w_neigh)
        return stats.pearsonr(weight_ij, weight_neighbor)
    
    def assign_weight_network(self):
        """
        만들어둔 correlated weight를 네트워크에 입력
        """
        key = list(self.edge_list_weight.keys())
        value = list(self.edge_list_weight.values())
        for i in range(self.lattice.number_of_edges()):
            self.lattice.edges[key[i]]['weight'] = value[i]
    
    def reconstruct_network(self):
        """
        만들어둔 correlated weight를 네트워크에 입력
        """
        self.lattice = nx.grid_2d_graph(self.size, self.size, periodic=self.Periodic)
        key = list(self.edge_list_weight.keys())
        value = list(self.edge_list_weight.values())
        for i in range(self.lattice.number_of_edges()):
            self.lattice.edges[key[i]]['weight'] = value[i]