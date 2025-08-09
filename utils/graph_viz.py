import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Paths (absolute)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ENRICHED_DIR = os.path.join(os.path.dirname(ROOT_DIR), "data", "enriched_subgraphs")

EDGE_TYPE_MAP = {
    0: "IS_CHILD_OF",
    1: "IS_PART_OF",
    2: "EXACT_MATCH",
    3: "INFERRED_ALIGNMENT",
}

def _norm01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return None

def render_graph_image(student_id, target_ccss):
    pt_path = os.path.join(ENRICHED_DIR, f"{target_ccss}.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Subgraph {target_ccss}.pt not found")

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    codes = data.code_strs
    scores = data.history_scores
    edge_index = data.edge_index
    edge_attr = (
        data.edge_attr.view(-1).tolist() if hasattr(data, "edge_attr") and data.edge_attr is not None
        else [0] * edge_index.size(1)
    )

    # Target index
    try:
        target_idx = codes.index(target_ccss)
    except ValueError:
        raise ValueError(f"{target_ccss} not found in its own subgraph!")

    # One-hop neighbourhood
    neighbor_idxs = {target_idx}
    for src, tgt in edge_index.t().tolist():
        if src == target_idx or tgt == target_idx:
            neighbor_idxs.update((src, tgt))

    # Build graph
    G = nx.Graph()
    for i in neighbor_idxs:
        label = codes[i]
        s = scores[i].get(student_id)
        s01 = _norm01(s)
        color = "#d3d3d3" if s01 is None else plt.cm.Reds(s01)
        G.add_node(i, label=label, color=color)

    for k, (src, tgt) in enumerate(edge_index.t().tolist()):
        if src in neighbor_idxs and tgt in neighbor_idxs:
            edge_label = EDGE_TYPE_MAP.get(edge_attr[k], "")
            G.add_edge(src, tgt, label=edge_label)

    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    node_labels = {n: G.nodes[n]["label"] for n in G.nodes}
    edge_labels = nx.get_edge_attributes(G, "label")

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(G, pos, ax=ax, with_labels=True, labels=node_labels,
            node_color=node_colors, edge_color="#888", font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    import base64 as _b64
    return _b64.b64encode(buf.read()).decode("utf-8")
