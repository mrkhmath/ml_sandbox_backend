# model/cache_loader.py
import os, time, requests, torch, threading

BASE_URL = "https://huggingface.co/datasets/mrkhmath/ccss-enriched-subgraphs/resolve/main/subgraphs"

# BASE_URL = os.environ.get(
#     "SUBGRAPH_BASE_URL",
#     "https://huggingface.co/datasets/mrkhmath/ccss-enriched-subgraphs/resolve/main/subgraphs"
# )
CACHE_DIR =  "/tmp/subgraphs"
# CACHE_DIR = os.environ.get("SUBGRAPH_CACHE", "/tmp/subgraphs")
MAX_CACHE_MB = 150
# MAX_CACHE_MB = int(os.environ.get("SUBGRAPH_MAX_CACHE_MB", "150"))
os.makedirs(CACHE_DIR, exist_ok=True)

# ---- NEW: concurrency guards ----
_download_locks = {}                   # per-code lock
_download_locks_mu = threading.Lock()
_dl_semaphore = 2 # max parallel downloads
# _dl_semaphore = threading.Semaphore(int(os.environ.get("MAX_PAR_DL", "2")))  # max parallel downloads

def _get_lock(code: str) -> threading.Lock:
    with _download_locks_mu:
        if code not in _download_locks:
            _download_locks[code] = threading.Lock()
        return _download_locks[code]

def _cache_bytes() -> int:
    total = 0
    for name in os.listdir(CACHE_DIR):
        p = os.path.join(CACHE_DIR, name)
        if os.path.isfile(p):
            try: total += os.path.getsize(p)
            except: pass
    return total

def _evict_if_needed(incoming_bytes: int):
    limit = MAX_CACHE_MB * 1024 * 1024
    size = _cache_bytes()
    if size + incoming_bytes <= limit:
        return
    files = []
    for name in os.listdir(CACHE_DIR):
        p = os.path.join(CACHE_DIR, name)
        if os.path.isfile(p):
            try: files.append((os.path.getatime(p), os.path.getsize(p), p))
            except: pass
    files.sort()  # LRU
    for _, sz, p in files:
        try:
            os.remove(p)
            size -= sz
            if size + incoming_bytes <= limit:
                break
        except: pass

def ensure_local(code: str):
    """Return (local_path, was_new). Only one thread downloads a given code."""
    if not BASE_URL:
        raise RuntimeError("SUBGRAPH_BASE_URL not set")
    local = os.path.join(CACHE_DIR, f"{code}.pt")
    if os.path.isfile(local):
        try: os.utime(local, None)
        except: pass
        return local, False

    lock = _get_lock(code)
    with lock:
        # re-check after acquiring lock
        if os.path.isfile(local):
            try: os.utime(local, None)
            except: pass
            return local, False

        url = f"{BASE_URL.rstrip('/')}/{code}.pt"
        tmp = local + ".part"
        print(f"[DL] {code} -> {url}", flush=True)

        _dl_semaphore.acquire()
        try:
            # simple retry/backoff
            delay = 0.8
            for attempt in range(3):
                try:
                    with requests.get(url, stream=True, timeout=45) as r:
                        if r.status_code == 404:
                            raise FileNotFoundError(f"Remote subgraph not found: {url}")
                        r.raise_for_status()
                        cl = r.headers.get("Content-Length")
                        incoming = int(cl) if cl and cl.isdigit() else 10 * 1024 * 1024
                        _evict_if_needed(incoming)

                        with open(tmp, "wb") as f:
                            for chunk in r.iter_content(1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                    os.replace(tmp, local)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(delay)
                    delay *= 1.8
        finally:
            _dl_semaphore.release()

        try: os.utime(local, None)
        except: pass
        return local, True

def load_subgraph(code: str):
    local, _ = ensure_local(code)
    return torch.load(local, map_location="cpu", weights_only=False)
