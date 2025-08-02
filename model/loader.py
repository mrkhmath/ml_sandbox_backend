import torch
import json
import os
from model.utils import generate_timeline_base64, generate_graph_base64
from torch.serialization import safe_globals

SEQUENCE_FILE = "data/student_sequences.json"
GRAPH_DIR = "data/pyg_subgraphs"

with open(SEQUENCE_FILE, "r") as f:
    student_sequences = json.load(f)

def build_sequence_and_metadata(student_id, target_ccss, dok, device):
    history = student_sequences.get(student_id, [])
    history_sorted = sorted(history, key=lambda x: int(x["assessment_seq"]))

    sequence = []
    for step in history_sorted:
        graph = torch.load(os.path.join(GRAPH_DIR, f"{step['canonical_ccss']}.pt"), weights_only=False)

        sequence.append({
            "graph": graph.to(device),
            "dok": torch.tensor([step["normalized_dok"]], dtype=torch.long, device=device)
        })

    safe_globals().update({
    "torch_geometric.data.data.DataEdgeAttr": __import__("torch_geometric").data.DataEdgeAttr})
    with safe_globals():
       graph = torch.load(os.path.join(GRAPH_DIR, f"{step['canonical_ccss']}.pt"), weights_only=False)
    sequence.append({
        "graph": graph.to(device),
        "dok": torch.tensor([dok], dtype=torch.long, device=device)
    })

    timeline_img = generate_timeline_base64(history_sorted, target_ccss)
    graph_img = generate_graph_base64(graph, target_ccss)

    return sequence, {"timeline_img": timeline_img, "graph_img": graph_img}

