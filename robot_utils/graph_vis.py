'''
For visualizing a design graph
'''

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def draw_graph(node_list, connection, show=True, save=None, lab=None, fitness=0):
    total_edge = 14
    total_node = 3
    node_list = np.array(node_list, dtype=int)
    node_types = node_list // total_edge
    edge_types = node_list % total_edge

    connection = np.array(connection, dtype=int)
    rand_color = np.random.uniform(size=[3, 3])
    rand_color[0] = [0, 0, 1]
    rand_color[1] = [1, 0, 0]
    rand_color[2] = [0, 1, 0]

    G = nx.Graph()
    color_map = []
    for i in range(len(node_list)):
        G.add_node(i)
        color_map.append(rand_color[node_types[i]])

    edge_color_map = []

    import colorsys

    colors = []
    for i in np.arange(0., 360., 360. / total_edge):
        hue = i / 360.
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    rand_edge_color = colors

    # a set of l/r edges
    rand_edge_color[11] = rand_edge_color[5]
    rand_edge_color[12] = rand_edge_color[6]
    rand_edge_color[13] = rand_edge_color[7]
    rand_edge_color[0] = rand_edge_color[1]
    rand_edge_color[8] = rand_edge_color[9]
    for i in range(len(node_list)):
        for j in range(i+1, len(node_list)):
            if connection[i,j] != 0:
                type_edge = edge_types[j]
                G.add_edge(i, j)
                edge_color_map.append(rand_edge_color[type_edge])
    pos = nx.spring_layout(G,seed=42)

    nx.draw(G, node_color=color_map, pos=pos, edge_color=edge_color_map)
    if save is not None:
        ax = plt.gca()
        plt.text(0.95, 0.95, lab, fontsize="xx-large",
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax.transAxes
                 )
        plt.text(0.95, 0.90, "score: "+str(fitness), fontsize="xx-large",
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax.transAxes,
                 color='red'
                 )
        plt.savefig(save)
        plt.clf()
    if show:
        plt.show()


if __name__ == '__main__':
    pass
