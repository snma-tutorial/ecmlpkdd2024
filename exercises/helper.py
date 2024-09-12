import pandas as pd
from netin.graphs import Graph
from netin.models import Model

from collections import Counter
from typing import List, Set, Union
import matplotlib.pyplot as plt
import numpy as np

from netin.utils import io
from netin.utils import constants

def get_title(df: pd.DataFrame, model_name: str, f_m: float, h_M: float, h_m: float) -> str:
    """
    Returns the name of the model including its parameters f_m, h_m, and h_M

    Parameters
    ----------
    df: pandas.DataFrame 
        where rows represent nodes, and columns their respective centrality scores

    model_name: str
        original model name (without parameters)

    f_m, h_m, h_M: float, float, float
        the model parameters to add in the title

    Returns
    -------
    title: str
    """
    g = [r"f$_{m}$=<fm>".replace("<fm>", f"{f_m}")]
    if h_M == h_m:
        s = r"h$_{M}$=h$_{m}$=<h>".replace("<h>", f"{h_M}")
    else:
        s = r"h$_{M}$>h$_{m}$" if h_M > h_m else r"h$_{M}$<h$_{m}$"
    g.append(s)
    return f"{model_name}\n{', '.join(g)}"

def get_edge_type_counts(graph:Graph, fraction:bool=False):
    """
    Returns the counts per edge type, e.g., How many edges between the minority and majority group (mM).

    Parameters
    ----------
    data: netin.models.Graph or List[netin.models.Graph] or Set[netin.models.Graph]
        a single graph or a list of graphs

    kwargs: dict
        width_bar, figsize, loc, nc_legend

    Returns
    -------
    counts: collections.Counter
    """
    edges = []
    for source, target in graph.edges():
        sc = graph.get_node_class(constants.CLASS_ATTRIBUTE).get_class_values()[source]
        tc = graph.get_node_class(constants.CLASS_ATTRIBUTE).get_class_values()[target]
        edges.append(f"{sc}{tc}")
    counts = Counter(edges)

    if fraction:
        total = sum(counts.values())
        counts = Counter({k: v / total for k, v in counts.items()})

    return counts


def plot_edge_type_counts(data: Union[Model, list[Model], set[Model]], fn=None, **kwargs):
    """
    Plots the edge type counts of a single or multiple graphs

    Parameters
    ----------
    data: netin.models.Model or List[netin.models.Model] or Set[netin.models.Model]
        a single model_graph or a list of model_graphs

    kwargs: dict
        width_bar, figsize, loc, nc_legend
    """

    if type(data) not in [list, set, List, Set]:
        data = [data]

    w, h = len(data) * 3.2, 3.2  # default figure size (width, height)
    width = kwargs.pop('width_bar', 0.25)  # the width of the bars
    figsize = kwargs.pop('figsize', (w, h))  # figure size (width, height)
    loc = kwargs.pop('loc', 'upper right')  # position of legend
    ncols = kwargs.pop('nc_legend', 1)  # number of columns in legend
    bbox_to_anchor = kwargs.pop('bbox_to_anchor', (0.5, -0.05))
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')
    multiplier = 0

    x = None
    groups = None
    maxy = 0
    for model_graph in data:
        etc = get_edge_type_counts(model_graph.graph) #g.calculate_edge_type_counts()
        name = f"{model_graph.SHORT}\nf_m={model_graph.f_m}, h_m={model_graph.h_m}, h_M={model_graph.h_M}"

        groups = list(etc.keys()) if groups is None else groups
        x = np.arange(len(groups)) if x is None else x
        y = [etc[i] for i in groups]
        maxy = max(max(y), maxy)

        offset = width * multiplier
        rects = ax.bar(x + offset, y, width, label=name)
        ax.bar_label(rects, padding=3)
        multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Counts')
        ax.set_title("Counts of edge types");
        ax.set_xticks(x + width, groups)
        ax.legend(loc=loc, ncols=ncols)

    # set limits
    ax.set_ylim(0, maxy * 1.1)

    # save plot
    if fn is not None:
        fig.savefig(fname=fn, bbox_inches='tight', dpi=300)
        print(f'{fn} saved.')

    # Final
    plt.show()
    plt.close()
