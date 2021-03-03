import numpy as np
import networkx as nx

class lattice_assign:
    def __init__(self, lattice_length=300, nodelist, edgelist):
        
        self.edges_list = edgelist
        self.nodes_list = nodelist
        self.length = lattice_length
        self.edge_speed_array = np.zeros(len(edgelist))
        self.edge_count = np.zeros(len(edgelist))
    
    def node_in_lattice(self, gps_point):
        """
        택시 gps를 입력받고, 이를 중심으로 한 length 길이의 격자 안에 네트워크 노드들 검색 
        """
        in_lattice = node_list[node_list['x'] <= gps_point[0] + self.length * 0.5]
        in_lattice = in_lattice[in_lattice['x'] >= gps_point[0] - self.length * 0.5]
        in_lattice = in_lattice[in_lattice['y'] >= gps_point[1] - self.length * 0.5]
        in_lattice = in_lattice[in_lattice['y'] <= gps_point[1] + self.length * 0.5]
        return list(in_lattice['osmid'])
    
    def assign_speed():
        """
        
        """
        pass