import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd

import unit_utils
import os

import scipy.io as sio
from scipy import stats
import pickle
import matplotlib as mpl

import seaborn as sns
from analysis import files
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
from scipy import stats
from scipy import signal


import networkx as nx
from itertools import combinations


# examine features of cells part of assemblies
# 1. label the cells + plot graphs with FR
# 2. firing properties (firing rate)
# 3. phase locking to ripples


sns.set_style('ticks')


assemblies = pickle.load(open('../output/assemblies_pre_template.pkl', 'rb'))

for strain in assemblies:

    for assembly in assemblies[strain]:
        session = assembly['session']
        if session != 'Hab Rec 1' and session != 'OLM Rec 1':
            continue


        templates = assembly['templates']
        num_templates = templates.shape[1]
        num_cells = templates.shape[0]
        cells = unit_utils.extract_cell_numbers(templates)
        unit_types = assembly['unit_types']
        #tt = assembly['tt']


        # get template reactivation strengths
        strengths = assembly['reactivation']

        # zscore
        z_strengths = stats.zscore(strengths, axis=1)
        avg_strength = np.mean(strengths, axis=1)

        activations = unit_utils.reactivation_indices(z_strengths)
        _, _, max_corr = unit_utils.activation_correlation(num_templates, z_strengths, activations,
                                                           assembly['ripple_bins'])

        # map between 1.0 and 5.0
        normalized = (5.0-1.0) * (avg_strength - np.min(avg_strength))/(np.max(avg_strength) - np.min(avg_strength)) + 1

        r_s = np.mean(z_strengths[:, assembly['ripple_bins']],axis=1)

        normalized_r = (2.0-0.5) * (r_s - np.min(r_s))/(np.max(r_s) - np.min(r_s)) + 0.5

        normalized_c = (2.0-0.5) * (max_corr - np.min(max_corr))/(np.max(max_corr) - np.min(max_corr)) + 0.5


        # plot sample to show clustering is good
        # plt.figure()
        # sns.heatmap(templates, cmap='icefire')
        # plt.show()
        # sns.heatmap(cells)
        """plt.figure()
        sns.heatmap(np.sum(z_matrix[:, activations[4]], axis=1).reshape(-1, 1))
        plt.show(block=False)
"""
        # networkx
        G = nx.Graph()
        for i in range(0, num_cells):
            G.add_node(i, unit_type=unit_types[i])#, tt=tt[i])


        for i in range(0, num_templates):
            c = np.nonzero(cells[:,i])[0]
            print(c)
            # for each pair add edge
            for pairs in combinations(c, 2):
                G.add_edge(pairs[0], pairs[1], assembly=i, strength=normalized[i], r_strength=normalized_c[i])

        isolated = list(nx.isolates(G))
        inter = np.where(unit_types == 1.0)[0].tolist()
        inter = [i for i in inter if i not in isolated]

        pyr = np.where(unit_types == 0.0)[0].tolist()
        pyr = [i for i in pyr if i not in isolated]
        #node_colors = list(nx.get_node_attributes(G, 'tt').values())
        #inode_colors = np.fromiter(nx.get_node_attributes(G, 'tt').values(), dtype=float)[inter]
        #pnode_colors = np.fromiter(nx.get_node_attributes(G, 'tt').values(), dtype=float)[pyr]

        G.remove_nodes_from(isolated)
        edge_colors = nx.get_edge_attributes(G, 'assembly').values()
        edge_strengths = list(nx.get_edge_attributes(G, 'strength').values())
        c_edge_strengths = list(nx.get_edge_attributes(G, 'r_strength').values())


        #node_shapes = list(map(lambda x: '^' if x == 1.0 else 'o', node_shapes))

        nodePos = nx.layout.spring_layout(G)


        # draw interneurons



        """
        plt.figure()
        nx.draw_networkx_nodes(G, pos=nodePos, nodelist=inter, node_shape='^')
        nx.draw_networkx_nodes(G, pos=nodePos, nodelist=pyr, node_shape='o')
        nx.draw_networkx_edges(G,nodePos, edge_color=edge_colors, width=edge_strengths, edge_cmap=mpl.colormaps['tab10'])
        plt.show()"""
        # separate based on cell type

        plt.figure()
        nx.draw_networkx_nodes(G, pos=nodePos, nodelist=inter, node_color='black', node_size=10, node_shape='^')
        nx.draw_networkx_nodes(G, pos=nodePos, nodelist=pyr, node_color='black', node_size=10, node_shape='o')
        nx.draw_networkx_edges(G, nodePos, edge_color=edge_colors, width=c_edge_strengths,
                               edge_cmap=mpl.colormaps['tab10'])
        plt.show()






