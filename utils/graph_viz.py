import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json
ENRICHED_DIR = "data/enriched_subgraphs"
HISTORY_PATH = "data/history_scores.json"

# Optional mapping of edge type IDs → names
EDGE_TYPE_MAP = {
    0: "IS_CHILD_OF",
    1: "IS_PART_OF",
    2: "EXACT_MATCH",
    3: "INFERRED_ALIGNMENT"
}

# Load full history once
with open(HISTORY_PATH) as f:
    history_scores = json.load(f)

def render_graph_image(student_id, target_ccss):
    pt_path = os.path.join(ENRICHED_DIR, f"{target_ccss}.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Subgraph {target_ccss}.pt not found")

    data = torch.load(pt_path)
    codes = data.code_strs
    scores = data.history_scores  # list of {student_id: score}
    edge_index = data.edge_index
    edge_attr = data.edge_attr.squeeze().tolist() if hasattr(data, "edge_attr") else [0] * edge_index.size(1)

    G = nx.Graph()
    node_map = {}

    # Find target node index
    try:
        target_idx = codes.index(target_ccss)
    except ValueError:
        raise ValueError(f"{target_ccss} not found in its own subgraph!")

    # Collect neighbor edges and nodes
    neighbor_idxs = set([target_idx])
    for i, (src, tgt) in enumerate(edge_index.t().tolist()):
        if src == target_idx or tgt == target_idx:
            neighbor_idxs.add(src)
            neighbor_idxs.add(tgt)

    # Add nodes with labels and colors
    for i in neighbor_idxs:
        label = codes[i]
        score = scores[i].get(student_id)
        color = "#d3d3d3" if score is None else plt.cm.Reds(score)
        G.add_node(i, label=label, color=color)

    # Add edges among visible nodes
    for i, (src, tgt) in enumerate(edge_index.t().tolist()):
        if src in neighbor_idxs and tgt in neighbor_idxs:
            edge_label = EDGE_TYPE_MAP.get(edge_attr[i], "")
            G.add_edge(src, tgt, label=edge_label)

    # Draw
    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    node_labels = {n: G.nodes[n]["label"] for n in G.nodes}
    edge_labels = nx.get_edge_attributes(G, "label")

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(G, pos, ax=ax, with_labels=True, labels=node_labels, node_color=node_colors, edge_color="#888", font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    # Convert to base64
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
