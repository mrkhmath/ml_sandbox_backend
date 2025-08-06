import os
import json
import torch
from torch.nn.functional import sigmoid
from model.load_model import load_trained_model

# === Paths ===
MODEL_PATH = "data/epoch14_acc0.7379_f10.7350.pt"
SEQ_PATH = "data/student_sequences.json"
SUBGRAPH_DIR = "data/enriched_subgraphs"

# === Load once ===
model = load_trained_model(MODEL_PATH)
model.eval()

with open(SEQ_PATH) as f:
    student_sequences = json.load(f)

def run_inference(student_id, target_ccss, normalized_dok):
    if student_id not in student_sequences:
        raise ValueError(f"Student {student_id} not found.")

    sequence = student_sequences[student_id]
    graph_sequence = []

    for entry in sequence:
        code = entry.get("canonical_ccss")
        dok = entry.get("normalized_dok", 1)

        pt_path = os.path.join(SUBGRAPH_DIR, f"{code}.pt")
        if not os.path.exists(pt_path):
            continue

        data = torch.load(pt_path)
        step = {
            "graph": data,
            "dok": torch.tensor([dok], dtype=torch.long)
        }
        graph_sequence.append(step)

    # Append target concept
    target_path = os.path.join(SUBGRAPH_DIR, f"{target_ccss}.pt")
    if not os.path.exists(target_path):
        raise ValueError(f"Target subgraph {target_ccss} not found.")

    target_graph = torch.load(target_path)
    graph_sequence.append({
        "graph": target_graph,
        "dok": torch.tensor([normalized_dok], dtype=torch.long)
    })

    # Run model
    with torch.no_grad():
        logits = model(graph_sequence)
        if isinstance(logits, torch.Tensor):
            prob = torch.sigmoid(logits[-1]).item()
            pred = int(prob >= 0.5)
        else:
            raise RuntimeError("Model output not tensor")

    return pred, prob
