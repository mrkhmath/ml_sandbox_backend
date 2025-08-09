# model/predict.py
import os
import json
from functools import lru_cache

import torch
from model.load_model import load_trained_model
from model.cache_loader import ensure_local, load_subgraph  # ensure_local returns (path, was_new)

# -------- Config --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "data", "epoch14_acc0.7379_f10.7350.pt")
SEQ_PATH   = os.path.join(ROOT_DIR, "data", "student_sequences.json")

THRESHOLD = 0.7
MAX_HISTORY_STEPS = 16       # cap history length
MAX_NEW_DL_PER_REQ = 8       # cap NEW downloads per request
# THRESHOLD = float(os.environ.get("READINESS_THRESHOLD", "0.7"))
# MAX_HISTORY_STEPS = int(os.environ.get("MAX_HISTORY_STEPS", "16"))        # cap history length
# MAX_NEW_DL_PER_REQ = int(os.environ.get("MAX_NEW_DL_PER_REQ", "8"))       # cap NEW downloads per request

# -------- Load once (CPU) --------
model = load_trained_model(MODEL_PATH)
model.to("cpu")
model.eval()

with open(SEQ_PATH, "r", encoding="utf-8") as f:
    student_sequences = json.load(f)

@lru_cache(maxsize=256)
def _cached_subgraph(code: str):
    # Normal load (may download on first call if not present)
    return load_subgraph(code)

def _tensor_dok(val) -> torch.Tensor:
    try:
        return torch.tensor([int(val)], dtype=torch.long)
    except Exception:
        return torch.tensor([1], dtype=torch.long)

def run_inference(student_id: str, target_ccss: str, normalized_dok: int):
    # ---- Validate ----
    if not isinstance(student_id, str) or not student_id:
        raise ValueError("Invalid student_id")
    if not isinstance(target_ccss, str) or not target_ccss:
        raise ValueError("Invalid target_ccss")
    if student_id not in student_sequences:
        raise ValueError(f"Student {student_id} not found.")

    seq = student_sequences[student_id]
    if MAX_HISTORY_STEPS > 0:
        seq = seq[-MAX_HISTORY_STEPS:]

    graph_sequence = []
    seen_codes = set()
    new_dl_count = 0

    # ---- Build history (bounded) ----
    for entry in seq:
        code = entry.get("canonical_ccss")
        if not code or code in seen_codes:
            continue
        seen_codes.add(code)

        # Only count when we actually need to download
        local_path, was_new = ensure_local(code)
        if was_new:
            new_dl_count += 1
            if new_dl_count > MAX_NEW_DL_PER_REQ:
                break  # stop adding more steps this request

        # Load from local (CPU) without bumping memory too much
        g = torch.load(local_path, map_location="cpu", weights_only=False)
        graph_sequence.append({
            "graph": g,
            "dok": _tensor_dok(entry.get("normalized_dok", 1)),
        })

    # ---- Append target (must exist) ----
    target_graph = _cached_subgraph(target_ccss)  # cached loader
    graph_sequence.append({
        "graph": target_graph,
        "dok": _tensor_dok(normalized_dok),
    })

    if not graph_sequence:
        raise RuntimeError("No graphs available for inference.")

    # ---- Model forward (CPU, no grad) ----
    with torch.no_grad():
        logits = model(graph_sequence)
        last = logits[-1] if isinstance(logits, (list, tuple)) else logits
        last = last.detach().to("cpu").reshape(-1)
        logit = last[-1] if last.numel() > 1 else last[0]
        prob = torch.sigmoid(logit).item()
        pred = int(prob >= THRESHOLD)

    return pred, prob
