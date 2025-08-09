# model/predict.py  (key lines)
MAX_HISTORY_STEPS = 20   # <- keep small
MAX_DOWNLOADS_PER_REQUEST = 25
# MAX_HISTORY_STEPS = int(os.getenv("MAX_HISTORY_STEPS", "20"))   # <- keep small
# MAX_DOWNLOADS_PER_REQUEST = int(os.getenv("MAX_DL_PER_REQ", "25"))

from functools import lru_cache
from model.cache_loader import load_subgraph
import torch, json, os
from model.load_model import load_trained_model

model = load_trained_model("data/epoch14_acc0.7379_f10.7350.pt")
model.eval()

with open("data/student_sequences.json","r",encoding="utf-8") as f:
    student_sequences = json.load(f)

@lru_cache(maxsize=256)
def _sg(code:str):
    return load_subgraph(code)

def run_inference(student_id, target_ccss, normalized_dok):
    seq = student_sequences.get(student_id)
    if not seq:
        raise ValueError(f"Student {student_id} not found.")

    # cap history to bound memory/IO
    seq = seq[-MAX_HISTORY_STEPS:] if MAX_HISTORY_STEPS > 0 else seq

    graph_sequence = []
    dl_count = 0
    seen = set()

    for entry in seq:
        code = entry.get("canonical_ccss")
        if not code or code in seen:
            continue
        seen.add(code)

        # hard cap on per-request downloads
        if dl_count >= MAX_DOWNLOADS_PER_REQUEST:
            break

        try:
            g = _sg(code)      # cached/remote
            dl_count += 1
            dok = int(entry.get("normalized_dok", 1))
            graph_sequence.append({"graph": g, "dok": torch.tensor([dok], dtype=torch.long)})
        except Exception:
            continue

    # target (must exist)
    tg = _sg(target_ccss)
    graph_sequence.append({"graph": tg, "dok": torch.tensor([int(normalized_dok)], dtype=torch.long)})

    with torch.no_grad():
        out = model(graph_sequence)
        last = out[-1] if isinstance(out, (list,tuple)) else out
        prob = torch.sigmoid(last.reshape(-1)[-1]).item()
        pred = int(prob >= 0.7)
    return pred, prob
