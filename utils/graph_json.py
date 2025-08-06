import os
import torch
import json

ENRICHED_DIR = "data/enriched_subgraphs"
HISTORY_PATH = "data/history_scores.json"
EDGE_TYPE_MAP = {
    0: "IS_CHILD_OF",
    1: "IS_PART_OF",
    2: "EXACT_MATCH",
    3: "INFERRED_ALIGNMENT"
}

with open(HISTORY_PATH) as f:
    history_scores = json.load(f)

def get_graph_json(student_id, target_ccss):
    pt_path = os.path.join(ENRICHED_DIR, f"{target_ccss}.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"{target_ccss}.pt not found")

    data = torch.load(pt_path)
    code_strs = data.code_strs
    edge_index = data.edge_index
    edge_types = data.edge_attr.squeeze().tolist() if hasattr(data, "edge_attr") else [0] * edge_index.size(1)
    scores = data.history_scores
    grades = data.grade_levels
    descriptions = data.descriptions

    target_idx = code_strs.index(target_ccss)
    visible_idxs = set([target_idx])

    for src, tgt in edge_index.t().tolist():
        if src == target_idx or tgt == target_idx:
            visible_idxs.update([src, tgt])

    # Nodes
    nodes = []
    for i in visible_idxs:
        nodes.append({
            "id": code_strs[i],
            "label": code_strs[i],
            "grade_levels": grades[i],
            "description": descriptions[i],
            "score": scores[i].get(student_id)
        })

    # Edges
    links = []
    for idx, (src, tgt) in enumerate(edge_index.t().tolist()):
        if src in visible_idxs and tgt in visible_idxs:
            links.append({
                "source": code_strs[src],
                "target": code_strs[tgt],
                "type": EDGE_TYPE_MAP.get(edge_types[idx], "RELATED")
            })

    return {"nodes": nodes, "links": links}
