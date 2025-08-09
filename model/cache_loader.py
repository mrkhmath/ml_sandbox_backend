import os, requests, torch

CACHE_DIR = os.environ.get("SUBGRAPH_CACHE", "/tmp/subgraphs")
BASE_URL  = os.environ.get("SUBGRAPH_BASE_URL", "")  # e.g. https://huggingface.co/datasets/USER/REPO/resolve/main/subgraphs

os.makedirs(CACHE_DIR, exist_ok=True)

def load_subgraph(code: str):
    local = os.path.join(CACHE_DIR, f"{code}.pt")
    if not os.path.isfile(local):
        if not BASE_URL:
            raise RuntimeError("SUBGRAPH_BASE_URL not set")
        url = f"{BASE_URL}/{code}.pt"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(local, "wb") as f:
            f.write(r.content)
    return torch.load(local, map_location="cpu", weights_only=False)
