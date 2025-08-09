import os, requests, torch

# CACHE_DIR = os.environ.get("SUBGRAPH_CACHE", "/tmp/subgraphs")
CACHE_DIR ="/tmp/subgraphs"
# BASE_URL  = os.environ.get("SUBGRAPH_BASE_URL")  # e.g. https://huggingface.co/datasets/mrkhmath/ccss-enriched-subgraphs/resolve/main/subgraphs
BASE_URL  = "https://huggingface.co/datasets/mrkhmath/ccss-enriched-subgraphs/resolve/main/subgraphs" # e.g. https://huggingface.co/datasets/mrkhmath/ccss-enriched-subgraphs/resolve/main/subgraphs
os.makedirs(CACHE_DIR, exist_ok=True)
# 
def load_subgraph(code: str):
    if not BASE_URL:
        raise RuntimeError("SUBGRAPH_BASE_URL not set")

    local = os.path.join(CACHE_DIR, f"{code}.pt")
    if not os.path.isfile(local):
        url = f"{BASE_URL.rstrip('/')}/{code}.pt"
        print(f"[DL] {code} -> {url}", flush=True)
        r = requests.get(url, timeout=45)
        if r.status_code == 404:
            raise FileNotFoundError(f"Remote subgraph not found: {url}")
        r.raise_for_status()
        with open(local, "wb") as f:
            f.write(r.content)
    return torch.load(local, map_location="cpu", weights_only=False)
