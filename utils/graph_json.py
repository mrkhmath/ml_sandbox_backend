import os
from functools import lru_cache
from model.cache_loader import load_subgraph

EDGE_TYPE_MAP = {
    0: "IS_CHILD_OF",
    1: "IS_PART_OF",
    2: "EXACT_MATCH",
    3: "INFERRED_ALIGNMENT",
}

@lru_cache(maxsize=256)
def _load(code: str):
    # cache per concept code; cache_loader handles download + torch.load(cpu)
    return load_subgraph(code)

def get_graph_json(student_id: str, target_ccss: str):
    # Load the target concept subgraph (remote + cached)
    data = _load(target_ccss)

    code_strs = data.code_strs
    edge_index = data.edge_index  # [2, E] tensor
    edge_types = (
        data.edge_attr.view(-1).tolist()
        if hasattr(data, "edge_attr") and data.edge_attr is not None
        else [0] * edge_index.size(1)
    )
    scores = getattr(data, "history_scores", None)
    grades = getattr(data, "grade_levels", None)
    descriptions = getattr(data, "descriptions", None)

    # Ensure target exists in this subgraph
    try:
        target_idx = code_strs.index(target_ccss)
    except ValueError:
        raise ValueError(f"Target code {target_ccss} not found in subgraph file.")

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
            "grade_levels": grades[i] if grades is not None else [],
            "description": descriptions[i] if descriptions is not None else "",
            "score": (scores[i].get(student_id) if scores is not None else None),
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
