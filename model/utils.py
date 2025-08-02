# utils.py
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from io import BytesIO
import base64

def encode_plt_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def generate_timeline_base64(sequence, target_ccss):
    labels = [s["canonical_ccss"] for s in sequence] + [target_ccss]
    readiness = [s["readiness"] for s in sequence] + [None]

    fig, ax = plt.subplots(figsize=(len(labels)*0.8, 2))
    ax.scatter(range(len(sequence)), [1]*len(sequence), c=readiness[:-1], cmap="coolwarm", s=200, edgecolors='k', label="Past")
    ax.scatter(len(sequence), 1, c="gold", s=250, edgecolors='k', label="Target")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_title("Assessment Timeline")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    return encode_plt_to_base64(fig)

def generate_graph_base64(graph_data, highlight_index):
    G = to_networkx(graph_data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42, k=0.15)

    fig, ax = plt.subplots(figsize=(5, 5))
    num_nodes = graph_data.num_nodes
    highlight_index = int(highlight_index)

    node_colors = ["orange" if i == highlight_index else "skyblue" for i in range(num_nodes)]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=80, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)

    if num_nodes <= 30:
        labels = {i: str(i) for i in range(num_nodes)}
        nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)

    ax.set_title("Concept Subgraph")
    ax.axis("off")
    plt.tight_layout()
    return encode_plt_to_base64(fig)
