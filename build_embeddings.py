"""
build_embeddings.py
Precompute LaBSE document embeddings and save to doc_embeddings.npz.

Run this once before starting the app:
    python build_embeddings.py

Memory-efficient: uses a numpy memmap so embeddings are written to disk
incrementally — the full matrix is never held in RAM.

At 400K docs (100K × EN/ES/FR/DE):
  - Encoding time  : ~1–2 hrs on Apple M-series (MPS)
  - Embedding file : ~1.2 GB  (400K × 768 × float32, compressed)
  - Peak RAM       : ~2–4 GB  (model + one batch + memmap window)

Author  : Thoshith S
"""

import os
import sys
import time
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LOKY_MAX_CPU_COUNT"]     = "1"
os.environ["OMP_NUM_THREADS"]        = "1"

CACHE_FILE      = "miracl_corpus_cache.jsonl"
META_FILE       = "miracl_corpus_cache_queries.json"
EMBEDDINGS_FILE = "doc_embeddings.npz"
PARTIAL_FILE    = "doc_embeddings_partial.npy"   # memmap during encoding
BATCH_SIZE      = 128   # larger batches → faster on MPS


def load_corpus_fast() -> tuple[list[str], list[str]]:
    """
    Read corpus from local JSONL cache.
    Returns (texts, doc_ids) — only the fields needed for encoding.
    """
    print(f"Reading corpus from {CACHE_FILE} …")
    texts   = []
    doc_ids = []
    with open(CACHE_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            texts.append(f"{d['title']}. {d['text'][:500]}")
            doc_ids.append(d["doc_id"])
    print(f"  {len(texts):,} documents loaded")
    return texts, doc_ids


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    if not os.path.exists(CACHE_FILE):
        print(f"ERROR: {CACHE_FILE} not found. Run corpus_loader.py first.")
        sys.exit(1)

    if os.path.exists(EMBEDDINGS_FILE):
        print(f"Embeddings already exist at {EMBEDDINGS_FILE}")
        ans = input("Re-encode? [y/N] ").strip().lower()
        if ans != "y":
            print("Skipping. Done.")
            return

    # ── Load corpus ───────────────────────────────────────────────────
    texts, doc_ids = load_corpus_fast()
    n = len(texts)

    # ── Load model ────────────────────────────────────────────────────
    MODEL_NAME = "sentence-transformers/LaBSE"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nLoading LaBSE on {device} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"  Model ready  |  {n:,} docs to encode  |  batch={BATCH_SIZE}\n")

    # ── Allocate memmap — writes directly to disk, no RAM accumulation ─
    size_mb = n * 768 * 4 / 1e6
    print(f"Allocating memmap {PARTIAL_FILE}  ({n:,} × 768 ≈ {size_mb:.0f} MB) …")
    mmap = np.memmap(PARTIAL_FILE, dtype="float32", mode="w+", shape=(n, 768))

    # ── Encode ────────────────────────────────────────────────────────
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, n, BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            enc = tokenizer(
                batch, max_length=128, padding=True,
                truncation=True, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out    = model(**enc)
            normed = F.normalize(out.pooler_output, p=2, dim=1)
            mmap[i: i + len(batch)] = normed.cpu().float().numpy()

            if i % 10_000 == 0 and i > 0:
                elapsed = time.time() - t0
                rate    = i / elapsed
                eta     = (n - i) / rate
                print(f"  {i:>8,}/{n:,}  ({100*i/n:.1f}%)  "
                      f"{rate:.0f} docs/s  ETA {eta/3600:.1f}h",
                      flush=True)

            if device == "mps" and i % 50_000 == 0:
                torch.mps.empty_cache()

    mmap.flush()
    del mmap   # close memmap

    # ── Convert memmap → compressed npz ──────────────────────────────
    print(f"\nSaving compressed {EMBEDDINGS_FILE} …")
    emb = np.memmap(PARTIAL_FILE, dtype="float32", mode="r", shape=(n, 768))
    doc_arr = np.array(doc_ids)
    np.savez_compressed(EMBEDDINGS_FILE, embeddings=emb, doc_ids=doc_arr)
    del emb

    os.remove(PARTIAL_FILE)

    elapsed = time.time() - t0
    size_gb = os.path.getsize(EMBEDDINGS_FILE) / 1e9
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/3600:.1f} h  ({elapsed/60:.0f} min)")
    print(f"Saved  → {EMBEDDINGS_FILE}  ({size_gb:.1f} GB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
