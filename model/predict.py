# model/predict.py
import json, os, torch
from functools import lru_cache
from model.load_model import load_trained_model
from model.cache_loader import load_subgraph  # ← uses HF URL + local cache

MODEL_PATH = "data/epoch14_acc0.7379_f10.7350.pt"
SEQ_PATH = "data/student_sequences.json"

model = load_trained_model(MODEL_PATH)
model.eval()

with open(SEQ_PATH, "r", encoding="utf-8") as f:
    student_sequences = json.load(f)

@lru_cache(maxsize=256)
def _sg(code: str):
    return load_subgraph(code)   # ← no os.path.exists

def run_inference(student_id, target_ccss, normalized_dok):
    if student_id not in student_sequences:
        raise ValueError(f"Student {student_id} not found.")

    seq = student_sequences[student_id]
    graph_sequence = []

    for entry in seq:
        code = entry.get("canonical_ccss")
        dok = int(entry.get("normalized_dok", 1))
        if not code:
            continue
        try:
            g = _sg(code)
            graph_sequence.append({"graph": g, "dok": torch.tensor([dok], dtype=torch.long)})
        except Exception:
            # silently skip missing codes in history
            continue

    # target (must exist)
    try:
        target_graph = _sg(target_ccss)
    except Exception:
        raise ValueError(f"Target subgraph {target_ccss} not found.")

    graph_sequence.append({"graph": target_graph, "dok": torch.tensor([int(normalized_dok)], dtype=torch.long)})

    with torch.no_grad():
        logits = model(graph_sequence)
        logit = logits[-1] if isinstance(logits, (list, tuple)) else logits
        prob = torch.sigmoid(logit.reshape(-1)[-1]).item()
        pred = int(prob >= 0.7)

    return pred, prob
