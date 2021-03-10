# test 20210310
import numpy as np
import networkx as nx

class lattice_assign:
    def __init__(self, lattice_length=300, nodelist, edgelist):
        
        self.edges_list = edgelist
        self.nodes_list = nodelist
        self.length = lattice_length
        self.edge_speed_array = np.zeros(len(edgelist))
        self.edge_count = np.zeros(len(edgelist))
    
    def node_in_lattice(self, taxi_data):
        """
        택시 gps 한 point 를 입력받고, 이를 중심으로 한 length 길이의 격자 안에 네트워크 노드들 검색 & 엣지 속도 어레이에 속도와 할당 수 입력 (self.edge_speed_array, self.edge_count)
        
        input
        ----------
            type : series 
                taxi data 
        """
        gps_point = taxi_data['x'], taxi_data['y']
        in_lattice = node_list[node_list['x'] <= gps_point[0] + self.length * 0.5]
        in_lattice = in_lattice[in_lattice['x'] >= gps_point[0] - self.length * 0.5]
        in_lattice = in_lattice[in_lattice['y'] >= gps_point[1] - self.length * 0.5]
        in_lattice = in_lattice[in_lattice['y'] <= gps_point[1] + self.length * 0.5]
        
        candidate_node_list = list(in_lattice['osmid'])
        candidate_edge_list = search_edge(candidate_node_list)
        assign_speed(candidate_edge_list)
        
    def assign_speed(self, taxi_speed, candidateEdgeList):
        """
        택시 gps 주위에 검색된 도로 네트워크 주변 링크에 햘당해줌
        self.edge_speed_array에 속도를 더해주고 self.edge_count에 할당 수 를 더해줌
        
        input
        ----------
            taxi_speed
                type : float
                taxi speed
            candidateEdgeList
                type : list            
        """
        for fid in candidateEdgeList:
            self.edge_speed_array[fid] += taxi_speed
            self.edge_count[fid] += 1

    
    def search_edge(self, candidateNodeList):
        """
        노드 리스트를 주면, 해당 노드들로 시작되거나 끝나는 링크 리스트를 검색해줌
        
        input
        ----------
            candidateNodeList
                type : list
                candiate node 'osmid' list
        return
        ----------
            candidate
                type : list 
                candidate edge 'fid' list
        """
        candidate_edge = []
        for node in candidateNodeList:
            u = self.edges_list[self.edges_list['u'] == node].index
            v = self.edges_list[self.edges_list['v'] == node].index
            candidate = candidate + list(u) + list(v)
        return candidate_edge