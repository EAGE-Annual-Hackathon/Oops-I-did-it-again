# import packages
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.options.display.max_colwidth = 2500
pd.set_option("display.max_columns", None)

import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import streamlit as st
from collections import OrderedDict
import networkx as nx
import yaml
import json
from itertools import combinations

import warnings

warnings.filterwarnings("ignore")

module_path = "../src/data/"
sys.path.append(os.path.abspath(module_path))
from config import get_config

module_path = "../src/logging/"
sys.path.append(os.path.abspath(module_path))
from logger import get_logger


# define logger
logger = get_logger()
logger.info("Logging started")

config = get_config(path="../data/config/", debug=False)

# define functions
def load_graph(is_directed=False) -> nx.classes.digraph.DiGraph:
    """
    Load graph data.
    """
    config = get_config(path="../data/config/", debug=False)

    path = f"../reports/graphs/graph_{'directed' if is_directed else 'undirected'}_{config['model_settings']['target_parameter']}.graphml"

    G = nx.read_graphml(path=path)

    return G


def load_coordinates() -> OrderedDict:
    """
    load locations of groups.
    """
    config = get_config(path="../data/config/", debug=False)

    file = f'../reports/graphs/positions_{config["model_settings"]["target_parameter"]}.json'

    with open(file, "r") as f:
        pos = json.loads(f.read(), object_pairs_hook=OrderedDict)

    return pos


def plot_directed_graph(DG, pos):
    """
    Plot directed graph.
    """

    config = get_config(path="../data/config/", debug=False)

    fig, ax = plt.subplots(figsize=(12, 12))

    # nodes
    node_size = 2000
    nodes = nx.draw_networkx_nodes(
        DG, pos, node_size=node_size, node_color="C2", alpha=0.25
    )
    nodes.set_zorder(0)

    # node labels
    nx.draw_networkx_labels(DG, pos, font_size=12, font_weight="bold", font_color="k")

    # edges
    threshold = 0.1

    edges_large = [
        (u, v) for (u, v, d) in DG.edges(data=True) if d["weight"] > threshold
    ]
    edges_small = [
        (u, v) for (u, v, d) in DG.edges(data=True) if d["weight"] <= threshold
    ]

    edges_color_large = pd.DataFrame(
        [(d) for (u, v, d) in DG.edges(data=True) if d["weight"] > threshold]
    )["weight"].values

    edges_color_small = pd.DataFrame(
        [(d) for (u, v, d) in DG.edges(data=True) if d["weight"] <= threshold]
    )["weight"].values

    width_as_weight = True
    multiplier = 6
    cmap = plt.cm.RdBu_r

    edges_large = nx.draw_networkx_edges(
        DG,
        pos,
        edgelist=edges_large,
        width=np.array(edges_color_large) * multiplier if width_as_weight else 4,
        alpha=1,
        arrowstyle="-|>",
        arrowsize=10,
        edge_color=edges_color_large,
        edge_cmap=cmap,
    )

    edges_small = nx.draw_networkx_edges(
        DG,
        pos,
        edgelist=edges_small,
        width=np.array(edges_color_small) * multiplier if width_as_weight else 2,
        alpha=0.5,
        style="dashed",
        arrowstyle="-|>",
        arrowsize=10,
        edge_color=edges_color_small,
        edge_cmap=cmap,
    )

    for edge_large, edge_small in zip(edges_large, edges_small):
        edge_large.set_zorder(1)
        edge_small.set_zorder(1)

    # edge weight labels
    edge_labels = nx.get_edge_attributes(DG, "weight")
    nx.draw_networkx_edge_labels(DG, pos, edge_labels=edge_labels)

    # colorbar
    vmin = min(edges_color_small)
    vmax = max(edges_color_large)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm)

    # decorations
    plt.title(f"{config['model_settings']['target_parameter']}")
    plt.axis("equal")
    plt.grid(ls="--")
    plt.xlabel("x")
    plt.ylabel("y")


def change_config(target_parameter: str) -> dict:
    """
    Change config file.
    """
    config = get_config(path="../data/config/", debug=False)
    config["model_settings"]["target_parameter"] = target_parameter

    # обновление конфига
    with open("../data/config/" + "config.yml", encoding="utf-8-sig", mode="w") as f:
        documents = yaml.dump(config, allow_unicode=True, stream=f)


def plot_undirected_graph(G, pos):
    """
    Plot undirected graph.
    """

    config = get_config(path="../data/config/", debug=False)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("xkcd:white")

    # nodes
    node_size = 200
    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=node_size, node_color="C3", alpha=1
    )

    nodes.set_zorder(2)

    # node labels
    nx.draw_networkx_labels(
        G, pos, font_size=12, font_weight="bold", font_color="k",
    )

    # edges
    edges = [(d["weight"]) for (u, v, d) in G.edges(data=True)]

    multiplier = 50
    edge_width = np.array(edges) * multiplier

    edges = nx.draw_networkx_edges(G, pos, edge_color="C0", alpha=0.2, width=edge_width)
    edges.set_zorder(0)

    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # decorations
    plt.title(f"{config['model_settings']['target_parameter']}")
    plt.axis("equal")
    plt.grid(ls="--")
    plt.xlabel("x")
    plt.ylabel("y")

    return fig


def main():
    page = st.sidebar.selectbox(
        "Select a Page",
        [
            "Homepage",
            "BORE_OIL_VOL",
            "BORE_GAS_VOL",
            "AVG_DOWNHOLE_PRESSURE",
            "BORE_WAT_VOL",
            "Graph comparison",
        ],
    )

    if page == "Homepage":
        homepage()

    if page == "BORE_OIL_VOL":
        oil("BORE_OIL_VOL")

    elif page == "BORE_GAS_VOL":
        gas("BORE_GAS_VOL")

    elif page == "AVG_DOWNHOLE_PRESSURE":
        pressure("AVG_DOWNHOLE_PRESSURE")

    elif page == "BORE_WAT_VOL":
        pressure("BORE_WAT_VOL")

    elif page == "Graph comparison":
        graph_comparison()


def homepage():
    st.write(
        """
        # EAGE Annual Hackathon 2022
        ## Opps, I dit it again
        Choose a page to explore the data."""
    )


def page_func(target):
    change_config(target)
    G = load_graph()
    pos = load_coordinates()
    fig = plot_undirected_graph(G, pos)

    st.pyplot(fig)


def oil(target):
    st.write(
        """
        # EAGE Annual Hackathon 2022
        ## Opps, I dit it again
        ### Graph
        """
    )

    page_func(target)


def gas(target):
    st.write(
        """
        # EAGE Annual Hackathon 2022
        ## Opps, I dit it again
        ### Graph
        """
    )
    page_func(target)


def pressure(target):
    st.write(
        """
        # EAGE Annual Hackathon 2022
        ## Opps, I dit it again
        ### Graph
        """
    )
    page_func(target)


def graph_comparison():
    st.write(
        """
        # EAGE Annual Hackathon 2022
        ## Opps, I dit it again
        ### Comparison of graphs using GED
        """
    )
    is_directed = True
    is_plot = False

    target_parameter = "BORE_OIL_VOL"
    change_config(target_parameter)
    DG_0 = load_graph(is_directed)
    pos_0 = load_coordinates()
    if is_plot:
        plot_directed_graph(DG_0, pos_0) if is_directed else plot_undirected_graph(
            DG_0, pos_0
        )

    target_parameter = "BORE_WAT_VOL"
    change_config(target_parameter)
    DG_1 = load_graph(is_directed)
    pos_1 = load_coordinates()
    if is_plot:
        plot_directed_graph(DG_1, pos_1) if is_directed else plot_undirected_graph(
            DG_1, pos_1
        )

    target_parameter = "AVG_DOWNHOLE_PRESSURE"
    change_config(target_parameter)
    DG_2 = load_graph(is_directed)
    pos_2 = load_coordinates()
    if is_plot:
        plot_directed_graph(DG_2, pos_2) if is_directed else plot_undirected_graph(
            DG_2, pos_2
        )

    target_parameter = "BORE_GAS_VOL"
    change_config(target_parameter)
    DG_3 = load_graph(is_directed)
    pos_3 = load_coordinates()
    if is_plot:
        plot_directed_graph(DG_3, pos_3) if is_directed else plot_undirected_graph(
            DG_3, pos_3
        )

    distances = np.array([])

    dictionary = {
        DG_0: "BORE_OIL_VOL",
        DG_1: "BORE_WAT_VOL",
        DG_2: "AVG_DOWNHOLE_PRESSURE",
        DG_3: "BORE_GAS_VOL",
    }
    for i in combinations(dictionary, 2):
        distances = np.append(distances, nx.graph_edit_distance(*i))

    data = pd.DataFrame(list(combinations({v: k for k, v in dictionary.items()}, 2)))
    data["distance"] = distances

    df = data.copy()
    df["1_"] = df[0]
    df["0_"] = df[1]
    df[1] = df["1_"]
    df[0] = df["0_"]
    df.drop(columns=["1_", "0_"], inplace=True)
    data = data.append(df)
    data

    def plot_heatmap(df: pd.DataFrame):
        """
        Plot heatmap.
        """

        fig = plt.figure(figsize=(10, 10))
        plt.title(f"Distance between graphs")

        sns.heatmap(
            df,
            xticklabels=df.columns,
            yticklabels=df.columns,
            annot=True,
            square=True,
            cmap="RdBu_r",
            cbar=True,
            fmt=".0f",
            cbar_kws={"label": "graph edit distance"},
        )

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        plt.xlabel(None)
        plt.ylabel(None)

        return fig

    df = data.pivot_table(index=0, columns=1, values="distance")

    fig = plot_heatmap(df)
    st.pyplot(fig)


if __name__ == "__main__":
    main()
