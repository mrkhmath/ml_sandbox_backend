import os
import torch
import json
from functools import lru_cache
from model.cache_loader import load_subgraph

@lru_cache(maxsize=256)
def _load_subgraph(code: str):
    return load_subgraph(code)

# Paths (absolute)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # this file's dir
ENRICHED_DIR = os.path.join(os.path.dirname(ROOT_DIR), "data", "enriched_subgraphs")

EDGE_TYPE_MAP = {
    0: "IS_CHILD_OF",
    1: "IS_PART_OF",
    2: "EXACT_MATCH",
    3: "INFERRED_ALIGNMENT",
}

def get_graph_json(student_id, target_ccss):
    pt_path = os.path.join(ENRICHED_DIR, f"{target_ccss}.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"{target_ccss}.pt not found")

    # Force CPU deserialisation; avoid scalar squeeze trap
    data = _load_subgraph(pt_path, map_location="cpu", weights_only=False)

    code_strs = data.code_strs
    edge_index = data.edge_index  # [2, E] tensor
    edge_types = (
        data.edge_attr.view(-1).tolist() if hasattr(data, "edge_attr") and data.edge_attr is not None
        else [0] * edge_index.size(1)
    )
    scores = data.history_scores  # list[dict], one per node
    grades = data.grade_levels
    descriptions = data.descriptions

    # Ensure target exists in this subgraph
    try:
        target_idx = code_strs.index(target_ccss)
    except ValueError:
        raise ValueError(f"Target code {target_ccss} not found in subgraph file {target_ccss}.pt")

    # One-hop neighbourhood of target
    visible_idxs = {target_idx}
    for src, tgt in edge_index.t().tolist():
        if src == target_idx or tgt == target_idx:
            visible_idxs.update((src, tgt))

    # Nodes
    nodes = []
    for i in sorted(visible_idxs):
        nodes.append({
            "id": code_strs[i],
            "label": code_strs[i],
            "grade_levels": grades[i],
            "description": descriptions[i],
            "score": scores[i].get(student_id),
        })

    # Edges (only among visible nodes)
    links = []
    for idx, (src, tgt) in enumerate(edge_index.t().tolist()):
        if src in visible_idxs and tgt in visible_idxs:
            links.append({
                "source": code_strs[src],
                "target": code_strs[tgt],
                "type": EDGE_TYPE_MAP.get(edge_types[idx], "RELATED"),
            })

    return {"nodes": nodes, "links": links}
