import os
import json
from functools import lru_cache
import torch
from model.load_model import load_trained_model

# === Paths (use absolute paths to be safe on Render) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # model/
ROOT_DIR = os.path.dirname(BASE_DIR)                   # project root
MODEL_PATH = os.path.join(ROOT_DIR, "data", "epoch14_acc0.7379_f10.7350.pt")
SEQ_PATH = os.path.join(ROOT_DIR, "data", "student_sequences.json")
SUBGRAPH_DIR = os.path.join(ROOT_DIR, "data", "enriched_subgraphs")

# === Tunables ===
MAX_HISTORY_STEPS = int(os.environ.get("MAX_HISTORY_STEPS", "60"))  # cap per-request I/O
THRESHOLD = float(os.environ.get("READINESS_THRESHOLD", "0.7"))

# === Load once (CPU) ===
model = load_trained_model(MODEL_PATH)
model.eval()
model.to("cpu")

with open(SEQ_PATH, "r", encoding="utf-8") as f:
    student_sequences = json.load(f)

def _subgraph_path(code: str) -> str:
    return os.path.join(SUBGRAPH_DIR, f"{code}.pt")

@lru_cache(maxsize=256)  # cache recent subgraphs to avoid repeated disk I/O
def _load_subgraph(code: str):
    path = _subgraph_path(code)
    if not os.path.isfile(path):
        return None
    # Force CPU deserialisation to avoid CUDA context lookups on Render free tier
    return torch.load(path, map_location="cpu", weights_only=False)

def run_inference(student_id: str, target_ccss: str, normalized_dok: int):
    # --- Validate inputs early (so Flask can return 400 instead of 502) ---
    if not isinstance(student_id, str) or not student_id:
        raise ValueError("Invalid student_id")
    if not isinstance(target_ccss, str) or not target_ccss:
        raise ValueError("Invalid target_ccss")
    if normalized_dok is None:
        raise ValueError("normalized_dok is required")
    try:
        normalized_dok = int(normalized_dok)
    except Exception:
        raise ValueError("normalized_dok must be an integer")

    if student_id not in student_sequences:
        raise ValueError(f"Student {student_id} not found.")

    # --- Build (capped) graph sequence ---
    sequence = student_sequences[student_id]
    graph_sequence = []

    # Use only the last N steps to keep requests fast and predictable
    recent_entries = sequence[-MAX_HISTORY_STEPS:] if MAX_HISTORY_STEPS > 0 else sequence

    for entry in recent_entries:
        code = entry.get("canonical_ccss")
        dok = int(entry.get("normalized_dok", 1))
        if not code:
            continue

        g = _load_subgraph(code)
        if g is None:
            continue

        step = {
            "graph": g,                             # PyG Data or your custom object
            "dok": torch.tensor([dok], dtype=torch.long)
        }
        graph_sequence.append(step)

    # --- Append target concept (must exist) ---
    target_graph = _load_subgraph(target_ccss)
    if target_graph is None:
        raise ValueError(f"Target subgraph {target_ccss} not found.")

    graph_sequence.append({
        "graph": target_graph,
        "dok": torch.tensor([normalized_dok], dtype=torch.long)
    })

    if not graph_sequence:
        raise RuntimeError("No graphs available for inference.")

    # --- Run model on CPU, no grad ---
    with torch.no_grad():
        # If your model expects CPU tensors, ensure any internal ops stay on CPU.
        logits = model(graph_sequence)

        # Support several shapes/containers safely
        if isinstance(logits, (list, tuple)) and len(logits) > 0:
            last_logit = logits[-1]
        else:
            last_logit = logits

        if not isinstance(last_logit, torch.Tensor):
            raise RuntimeError("Model output is not a tensor")

        last_logit = last_logit.detach().to("cpu").reshape(-1)
        # If the model returns a single logit, take index 0
        logit = last_logit[-1] if last_logit.numel() > 1 else last_logit[0]
        prob = torch.sigmoid(logit).item()

        pred = int(prob >= THRESHOLD)

    return pred, prob
