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

    fig, ax = plt.subplots(figsize=(len(labels)*0.7, 1.8))
    ax.scatter(range(len(sequence)), [1]*len(sequence), c=readiness[:-1], cmap="coolwarm", s=160, edgecolors='k')
    ax.scatter(len(sequence), 1, c="yellow", s=200, edgecolors='k', label="Target")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks([])
    ax.set_title("Assessment Timeline")
    plt.tight_layout()
    return encode_plt_to_base64(fig)

def generate_graph_base64(graph_data, highlight_code):
    G = to_networkx(graph_data, to_undirected=True)
    fig, ax = plt.subplots(figsize=(3,3))
    node_colors = ["orange" if str(i)==highlight_code else "skyblue" for i in range(graph_data.num_nodes)]
    nx.draw(G, with_labels=False, node_color=node_colors, node_size=150, ax=ax)
    plt.title("Concept Subgraph")
    plt.tight_layout()
    return encode_plt_to_base64(fig)
