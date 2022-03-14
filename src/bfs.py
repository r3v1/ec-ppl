from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import tikzplotlib
from networkx.drawing.nx_pydot import graphviz_layout


def plot_tree(tree: nx.Graph,
              figsize=(19.2 / 2, 10.8 / 2),
              title: str = "",
              savefig: Path = None,
              labels: dict = None,
              show: bool = False,
              savetikz: bool = False):
    assert type(tree) == nx.Graph, f"Tree object not a nx.Graph"
    assert len(tree) > 0, f"Empty graph"

    names = {
        "Bernoulli": "$Bernoulli$",
        "B": "$B$",
        "Be": "$Be$",
        # "Chi": r"$\chi^2$",
        "Chi": r"$Chi2$",
        "Exp": "$Exp$",
        # "N": r"$\mathcal{N}$",
        # "U": r"$\mathcal{U}$",
        "N": r"$N$",
        "U": r"$U$",
        "Pois": "$Pois$",
        "Binomial": "$Binomial$",
    }

    with plt.style.context(['science', 'ieee']):
        _, ax = plt.subplots(1, 1, figsize=figsize)

        # Define acronyms for long function names
        for k, v in labels.items():
            v = v.replace("Real", "R").replace("Natural", "N")
            v = v.replace("Positive", "P")
            v = v.replace("Tensor", "T")

            labels[k] = v

        pos = graphviz_layout(tree, prog="dot")
        x_pos = {i for i, _ in pos.values()}
        y_pos = sorted(list({j for _, j in pos.values()}))
        y_pos = [(y_pos[i] + y_pos[i + 1]) / 2 for i in range(len(y_pos) - 1)]

        # Distribution for each level
        # distributions_levels = {k: v for k, v in labels.items() if v[1:-1] in names.keys()}
        # y_pos = np.array(sorted(list({j for i, (_, j) in pos.items() if i in distributions_levels.keys()})))
        # y_distance = np.diff(sorted(list({j for i, (_, j) in pos.items()}))).mean() / 2
        # y_pos -= y_distance

        # node_colors = ["lightblue" if not n.startswith("0.") else "lightgreen" for n in tree.nodes()]
        nx.draw(
            tree,
            pos=pos,
            with_labels=False,
            node_size=750,
            width=0.5,
            ax=ax,
            node_shape='s',
            node_color="none"
        )

        labels = labels or {i: names[i.split(".")[-1]] for i in tree}
        nx.draw_networkx_labels(tree,
                                pos,
                                labels,
                                font_size=10,
                                ax=ax,
                                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.75')
                                )

        plt.hlines(y_pos, min(x_pos) - 50, max(x_pos) + 50, lw=0.5, colors="gray", linestyles='--')

        for level, y in enumerate(y_pos[::-1], start=1):
            plt.text(min(x_pos) - 50, y + 5, rf"$Level\ {level}$")

        plt.title(title, fontsize=14)
        plt.tight_layout()

        if savetikz and savefig is not None:
            tikzfile = savefig.parent / savefig.name.replace(".png", ".tex")
            phi = 1.618033988
            scale = 0.75
            tikzplotlib.save(tikzfile.as_posix(),
                             # extra_axis_parameters=["dashed"],
                             axis_width=f'{scale:.3}\linewidth',
                             axis_height=f'{scale / phi:.3}\linewidth')

        if savefig:
            plt.savefig(savefig, dpi=128)

        if show:
            plt.show()

    plt.close()


def simplify(nodes, edges, labels) -> Tuple[set, list, dict]:
    # Locate intermediate nodes
    inter = [k for k, v in labels.items() if "to" in v]
    new_labels = {k: v for k, v in labels.items() if "to" not in v}
    new_nodes = set(nodes) - set(inter)

    new_edges = []
    i = None
    o = None
    for n in inter:
        for (i_, o_) in filter(lambda e: n in e, edges):
            if i_ not in inter:
                i = i_

            if o_ not in inter:
                o = o_

            if i is not None and o is not None:
                new_edges.append((i, o))
                i = None
                o = None

    for n in new_nodes:
        for (i_, o_) in filter(lambda e: n in e, edges):
            if i_ not in inter and o_ not in inter:
                new_edges.append((i_, o_))

    return new_nodes, new_edges, new_labels


if __name__ == '__main__':
    # distribuciones = np.array(["Bernoulli", "B", "Be", "Chi", "Exp", "N", "U", "Pois"])
    # df = pd.DataFrame([
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0],
    #     [1, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 1, 1, 0, 1],
    #     [0, 0, 1, 0, 0, 1, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 1, 0],
    #     [1, 1, 0, 0, 0, 1, 0, 0],
    #     [0, 1, 0, 1, 0, 0, 0, 0]
    # ], columns=distribuciones, index=distribuciones)
    # i, j = np.where(df.values == 1)
    # edges = list(zip(list(distribuciones[i]), list(distribuciones[j])))
    #
    # G = nx.DiGraph()
    # G.add_nodes_from(distribuciones)
    #
    # G.add_edges_from(edges)
    # G = G.reverse()
    #
    # root = "Bernoulli"
    # depth = 4
    # bfs_tree = bfs_trees(G, root, depth)
    #
    # t = make_tree(bfs_tree)
    # plot_tree(t,
    #           figsize=(10, 5),
    #           title=rf"$X \sim {root}\ with\ {depth}\ levels$",
    #           savefig=Path(".") / f"{root}_{depth}.png")

    from utils import expr2individual
    from deap import gp

    expr = 'Binomial(toNatural0Tensor(toNaturalTensor(add(toTensor(x), toNatural0Tensor(toNaturalTensor(add(sub(toTensor(z), toTensor(z)), toNatural0Tensor(sub(toTensor(y), toTensor(x))))))))), toReal01Tensor(toRealPositiveTensor(toRealPositiveTensor(safeDiv(add(toTensor(y), add(toTensor(y), toTensor(y))), toTensor(z))))))'

    individual = expr2individual(expr)

    nodes, edges, labels = gp.graph(individual)

    nodes, edges, labels = simplify(nodes, edges, labels)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    plot_tree(
        g,
        title=rf"$Model$",
        labels={k: f"${v}$" for k, v in labels.items()},
        # savefig=working_dir / f"custom_model.png",
        show=True
    )
