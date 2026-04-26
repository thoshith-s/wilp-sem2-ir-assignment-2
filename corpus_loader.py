"""
corpus_loader.py
Loads the MIRACL multilingual corpus from HuggingFace Hub.

Primary source : thoshiths/miracl-multilingual-4M  (4M passages)
Evaluation queries are loaded from the official MIRACL dev qrels.
Supports EN, ES, FR, DE — 100K documents per language (400K total).

Author  : Thoshith S
"""

import json
import os
import random

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
REPO_ID   = "thoshiths/miracl-multilingual-4M"

LANGUAGES        = ["en", "es", "fr", "de"]
QUERIES_PER_LANG = {"en": 10, "es": 5, "fr": 5, "de": 5}
RANDOM_SEED      = 42
CACHE_FILE       = "miracl_corpus_cache.jsonl"
META_FILE        = "miracl_corpus_cache_queries.json"

def _resolve(filename: str) -> str:
    """Return /data/<filename> if mounted (HF Spaces bucket), else local path."""
    data_path = os.path.join("/data", filename)
    return data_path if os.path.exists(data_path) else filename


# ── Eval query helpers ─────────────────────────────────────────────────────────

def _load_topics(lang: str) -> dict:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        "miracl/miracl",
        f"miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-dev.tsv",
        repo_type="dataset", token=HF_TOKEN,
    )
    topics = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 2:
                topics[p[0]] = p[1]
    return topics


def _load_qrels(lang: str) -> dict:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        "miracl/miracl",
        f"miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-dev.tsv",
        repo_type="dataset", token=HF_TOKEN,
    )
    qrels: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 4 and int(p[3]) > 0:
                qrels.setdefault(p[0], []).append(p[2])
    return qrels


def _select_queries(lang: str, topics: dict, qrels: dict, n: int) -> list:
    valid = [(qid, topics[qid], docs) for qid, docs in qrels.items()
             if qid in topics and docs]
    random.seed(RANDOM_SEED)
    random.shuffle(valid)
    return [
        {
            "query_id":         f"{lang.upper()}{i+1:02d}",
            "query":            q,
            "language":         lang,
            "relevant_doc_ids": docs,
            "original_qid":     qid,
        }
        for i, (qid, q, docs) in enumerate(valid[:n])
    ]


# ── Load eval queries ──────────────────────────────────────────────────────────

def _build_queries() -> list:
    queries = []
    for lang in LANGUAGES:
        print(f"  [{lang}] loading topics & qrels …")
        topics   = _load_topics(lang)
        qrels    = _load_qrels(lang)
        selected = _select_queries(lang, topics, qrels, QUERIES_PER_LANG[lang])
        queries.extend(selected)
        print(f"    {len(selected)} queries selected")
    return queries


# ── Corpus loading ─────────────────────────────────────────────────────────────

DOCS_PER_LANG = 100_000   # subset per language from the 4M HF dataset

def _load_from_hub() -> list:
    """Load DOCS_PER_LANG docs per language from thoshiths/miracl-multilingual-4M."""
    from datasets import load_dataset

    print(f"Loading corpus from {REPO_ID}  ({DOCS_PER_LANG:,}/lang) …")
    ds = load_dataset(REPO_ID, split="train", token=HF_TOKEN)

    corpus = []
    for lang in LANGUAGES:
        lang_ds = ds.filter(lambda x: x["language"] == lang)
        n = min(DOCS_PER_LANG, len(lang_ds))
        for row in lang_ds.select(range(n)):
            corpus.append({
                "doc_id":       row["doc_id"],
                "title":        row["title"],
                "text":         row["text"],
                "language":     row["language"],
                "url":          row["url"],
                "miracl_docid": row["miracl_docid"],
            })
        print(f"  [{lang}] {n:,} docs")
    print(f"  Total: {len(corpus):,} docs")
    return corpus


# ── Public API ─────────────────────────────────────────────────────────────────

def load_miracl(cache_file: str = None,
                force_reload: bool = False) -> tuple[list, list]:
    """
    Load corpus + eval queries.

    Checks /data/ (HF Spaces bucket) before falling back to local files.
    Falls back to downloading from thoshiths/miracl-multilingual-4M if neither exists.
    """
    if cache_file is None:
        cache_file = _resolve(CACHE_FILE)
    meta_file = _resolve(META_FILE)

    if not force_reload and os.path.exists(cache_file) and os.path.exists(meta_file):
        print(f"Loading from cache: {cache_file}")
        corpus = _read_jsonl(cache_file)
        with open(meta_file, encoding="utf-8") as f:
            queries = json.load(f)
        _print_stats(corpus)
        return corpus, queries

    # ── Load corpus from Hub ──────────────────────────────────────────
    corpus = _load_from_hub()

    # ── Build eval queries ────────────────────────────────────────────
    print("\nLoading evaluation queries …")
    queries = _build_queries()

    # ── Map miracl_docid → doc_id for evaluator ───────────────────────
    miracl_to_corpus = {d["miracl_docid"]: d["doc_id"] for d in corpus}
    for q in queries:
        q["relevant_corpus_ids"] = [
            miracl_to_corpus[mid]
            for mid in q["relevant_doc_ids"]
            if mid in miracl_to_corpus
        ]

    # ── Cache locally ─────────────────────────────────────────────────
    print(f"\nWriting local cache → {cache_file} …")
    with open(cache_file, "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    print(f"Cached {len(corpus):,} docs → {cache_file}")
    print(f"Cached {len(queries)} queries → {meta_file}")

    _print_stats(corpus)
    return corpus, queries


# ── Fast JSONL reader ──────────────────────────────────────────────────────────

def _read_jsonl(path: str) -> list:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


# ── Backward-compat aliases ────────────────────────────────────────────────────

def load_corpus(cache_file: str = None, force_reload: bool = False) -> list:
    corpus, _ = load_miracl(cache_file=cache_file, force_reload=force_reload)
    return corpus


def load_queries(cache_file: str = META_FILE) -> list:
    with open(cache_file, encoding="utf-8") as f:
        return json.load(f)


# ── Stats ──────────────────────────────────────────────────────────────────────

def _print_stats(corpus: list) -> None:
    by_lang: dict[str, int] = {}
    for d in corpus:
        by_lang[d["language"]] = by_lang.get(d["language"], 0) + 1
    total   = len(corpus)
    en      = by_lang.get("en", 0)
    non_en  = total - en
    avg_len = sum(len(d["text"]) for d in corpus) / max(total, 1)
    print(f"\nCorpus statistics:")
    print(f"  Total documents : {total:,}")
    for lang in LANGUAGES:
        n = by_lang.get(lang, 0)
        print(f"  {lang.upper():5s}          : {n:>9,}  ({100*n/max(total,1):.1f}%)")
    print(f"  Non-English     : {non_en:,}  ({100*non_en/max(total,1):.1f}%)  ✅")
    print(f"  Avg text length : {avg_len:.0f} chars")


def get_corpus_stats(corpus: list) -> dict:
    by_lang: dict[str, int] = {}
    for d in corpus:
        by_lang[d["language"]] = by_lang.get(d["language"], 0) + 1
    en = by_lang.get("en", 0)
    return {
        "total":      len(corpus),
        "by_lang":    by_lang,
        "en_count":   en,
        "non_en_pct": 100 * (len(corpus) - en) / max(len(corpus), 1),
        "avg_length": sum(len(d["text"]) for d in corpus) / max(len(corpus), 1),
    }


if __name__ == "__main__":
    corpus, queries = load_miracl(force_reload=False)
    print(f"\nEval queries: {len(queries)}")
    for q in queries[:3]:
        print(f"  [{q['language']}] {q['query_id']}: {q['query'][:60]}")
        print(f"    relevant_corpus_ids: {q.get('relevant_corpus_ids', [])[:2]}")
