import networkx as nx
import numpy as np
import pandas as pd
from grant import grant
from grant.mtist.ground_truths.utils import (
    calc_midpoint,
    color_interactions,
    get_coef_color,
    get_interaction_pair_idxs_unique,
    judge_coefficients,
    judge_interactions,
)
from matplotlib import pyplot as plt


def graph_network(aij, ax=None, legend=True):

    if ax is None:
        fig, ax = plt.subplots()

    n_species = aij.shape[0]

    edgelist = generate_edgelist(n_species)  # list of pairs in sp notation
    int_types = judge_interactions(aij)  # list of int types in edgelist order
    colors = color_interactions(int_types)
    coef_pairs_independently = judge_coefficients(aij)  # list of coef types in edgelist order

    G = nx.Graph(edgelist)
    # pos = nx.spring_layout(G, k=1/np.sqrt(n_species) / 2, seed=3113794652)
    pos = nx.spring_layout(G, seed=3113794652)
    # pos = nx.kamada_kawai_layout(G)

    for edge, color, int_type in zip(edgelist, colors, int_types):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=3, ax=ax, edge_color=color)

    # pos = {0: (0, 0), 1: (0.5, 1), 2: (1, 0)}
    # labels = {0: 'species_0', 1: 'species_1', 2: 'species_2'}

    # node_options = {
    #     "font_size": 18,
    #     "node_size": 500,
    #     "node_color": "white",
    #     "edgecolors": "black",
    #     "linewidths": 2,
    #     "width": 2,
    # }

    nx.draw_networkx_nodes(
        G,
        pos,
        edgecolors="black",
        node_color="white",
        linewidths=2,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        # labels=labels,
        font_weight="bold",
        ax=ax,
    )

    # Now, to annotate all of the coefs themselves

    # fmt: off

    formatting_adjustments = {
        (1, 0): ((-0.15, +0.08), (-0.15, -0.15)),
        (2, 0): ((-0.5, 0.08), (-0.4, -0.08)),
        (2, 1): ((+0.1, +0.08), (+0.1, -0.08)),
    }

    strings = {
        (1, 0): (f"{coef_pairs_independently[0][1]} ->", f"<- {coef_pairs_independently[0][0]}"),
        (2, 0): (f"^ {coef_pairs_independently[1][0]}", f"v {coef_pairs_independently[1][1]}"),
        (2, 1): (f"^ {coef_pairs_independently[2][1]}", f"v {coef_pairs_independently[2][0]}"),
    }

    coef_colors = {
        (1, 0): (get_coef_color(coef_pairs_independently[0][1]), get_coef_color(coef_pairs_independently[0][0])),
        (2, 0): (get_coef_color(coef_pairs_independently[1][0]), get_coef_color(coef_pairs_independently[1][1])),
        (2, 1): (get_coef_color(coef_pairs_independently[2][1]), get_coef_color(coef_pairs_independently[2][0])),
    }

    # fmt: on

    adjs = formatting_adjustments

    for a_number, edge in enumerate(edgelist):

        node1 = edge[0]
        node2 = edge[1]

        midpoint = calc_midpoint(pos[node1], pos[node2])

        ax.text(
            midpoint[0] + adjs[edge][0][0],
            midpoint[1] + adjs[edge][0][1],
            strings[edge][0],
            color=coef_colors[edge][0],
            fontweight="bold",
        )

        ax.text(
            midpoint[0] + adjs[edge][1][0],
            midpoint[1] + adjs[edge][1][1],
            strings[edge][1],
            color=coef_colors[edge][1],
            fontweight="bold",
        )

    ### Finish off the fig

    ax.axis("off")

    if legend:
        n_unique_entries = len(list(pd.Series(int_types).unique()))
        int_names_for_legend = list(pd.Series(int_types).unique())
        colors_for_legend = list(pd.Series(colors).unique())

        grant.custom_legend(
            n_unique_entries,
            int_names_for_legend,
            color_array=colors_for_legend,
            ax=ax,
            marker=None,
            linestyle="-",
        )


def generate_edgelist(n_species):

    pairs = get_interaction_pair_idxs_unique(n_species)
    matrix_of_interactions = generate_matrix_of_interactions(n_species)
    edgelist = []
    for coef1, _ in pairs:
        edgelist.append(tuple(int(each) for each in matrix_of_interactions[coef1].split(",")))
    return edgelist


def generate_matrix_of_interactions(n_species):
    matrix_of_interactions = np.zeros((n_species, n_species), dtype=object)
    for i in range(3):
        for j in range(3):

            first_glyph = max(i, j)
            second_glyph = min(i, j)
            matrix_of_interactions[i, j] = f"{first_glyph},{second_glyph}"

    return matrix_of_interactions
