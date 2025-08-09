# model/cache_loader.py
import os, time, requests, torch

# BASE_URL = os.environ.get(
    # "SUBGRAPH_BASE_URL",
BASE_URL = "https://huggingface.co/datasets/mrkhmath/ccss-enriched-subgraphs/resolve/main/subgraphs"
# )
# CACHE_DIR = os.environ.get("SUBGRAPH_CACHE", "/tmp/subgraphs")
CACHE_DIR =  "/tmp/subgraphs"
# MAX_CACHE_MB = int(os.environ.get("SUBGRAPH_MAX_CACHE_MB", "200"))  # cap total cache size
MAX_CACHE_MB = 200  # cap total cache size
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_bytes() -> int:
    return sum((os.path.getsize(os.path.join(CACHE_DIR, f))
               for f in os.listdir(CACHE_DIR)
               if os.path.isfile(os.path.join(CACHE_DIR, f))), 0)

def _evict_if_needed(incoming_bytes: int):
    limit = MAX_CACHE_MB * 1024 * 1024
    size = _cache_bytes()
    if size + incoming_bytes <= limit:
        return
    # Evict least-recently-used files by atime
    files = []
    for name in os.listdir(CACHE_DIR):
        p = os.path.join(CACHE_DIR, name)
        if os.path.isfile(p):
            files.append((os.path.getatime(p), os.path.getsize(p), p))
    files.sort()  # oldest access first
    for _, sz, p in files:
        try:
            os.remove(p)
            size -= sz
            if size + incoming_bytes <= limit:
                break
        except Exception:
            pass

def load_subgraph(code: str):
    if not BASE_URL:
        raise RuntimeError("SUBGRAPH_BASE_URL not set")

    local = os.path.join(CACHE_DIR, f"{code}.pt")
    if not os.path.isfile(local):
        url = f"{BASE_URL.rstrip('/')}/{code}.pt"
        tmp = local + ".part"
        print(f"[DL] {code} -> {url}", flush=True)

        with requests.get(url, stream=True, timeout=45) as r:
            if r.status_code == 404:
                raise FileNotFoundError(f"Remote subgraph not found: {url}")
            r.raise_for_status()
            # estimate size for eviction (falls back if unknown)
            cl = r.headers.get("Content-Length")
            incoming = int(cl) if cl and cl.isdigit() else 10 * 1024 * 1024
            _evict_if_needed(incoming)

            # stream to disk to avoid big memory spikes
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp, local)  # atomic move

    # touch atime for LRU
    try:
        now = time.time()
        os.utime(local, (now, now))
    except Exception:
        pass

    return torch.load(local, map_location="cpu", weights_only=False)
