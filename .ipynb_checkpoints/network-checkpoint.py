import numpy as np
import pandas as pd
import networkx as nx

class traffic_Network:
    def __ini__(self, G):
        self.network = G
        
    def neighbor_link_weight(self, edge, edge_weight='weight'):
        """
        입력된 edge에 연결된 링크 정보를 가져옴. 
        input
        ----------
            type : tuple
            (u, v)
        return
        ----------
            typpe : list
            [(u, v, weight), ...]
        """
        neighbor_link = list(self.network.in_edges(edge[0], data=edge_weight))\
                    +list(self.network.out_edges(edge[0], data=edge_weight))\
                    +list(self.network.in_edges(edge[1], data=edge_weight))\
                    +list(self.network.out_edges(edge[1], data=edge_weight))
        # 자기 자신이 두번 불러와지므로, 2번 제거
        link = (edge[0], edge[1], self.network.get_edge_data(edge[0], edge[1],)[0][edge_weight])
        for i in range(2):neighbor_link.remove(input_link)
        return neighbor_link
    
    def mean_wegiht_neighbor(self, edge, edge_weight='weight'):
        """
        입력된 edge에 연결된 링크들의 weight의 평균
        """
        return np.mean(np.array(neighbor_link_weight(edge, edge_weight)[:,2]))
    
    