# evaluator.py
"""
Evaluation against the official MIRACL dev-set relevance judgments.
Covers 25 queries: 10 EN + 5 ES + 5 FR + 5 DE.
Computes: Precision@k, Recall@k, F1@k, Average Precision (AP), MAP, nDCG@k.

Author  : Thoshith S
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from corpus_loader import load_queries


# ── Lazy-load eval queries from cache ─────────────────────────────────────────
try:
    EVALUATION_QUERIES = load_queries()
except FileNotFoundError:
    EVALUATION_QUERIES = []   # populated after first load_miracl() call


# ── Metrics ────────────────────────────────────────────────────────────────────

class IRMetrics:

    @staticmethod
    def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        top_k = retrieved[:k]
        hits  = sum(1 for d in top_k if d in relevant)
        return hits / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        hits  = sum(1 for d in top_k if d in relevant)
        return hits / len(relevant)

    @staticmethod
    def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        p = IRMetrics.precision_at_k(retrieved, relevant, k)
        r = IRMetrics.recall_at_k(retrieved, relevant, k)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @staticmethod
    def average_precision(retrieved: list[str], relevant: set[str]) -> float:
        if not relevant:
            return 0.0
        hits = ap = 0.0
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                hits += 1
                ap   += hits / i
        return ap / len(relevant)

    @staticmethod
    def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        dcg = sum(
            1 / math.log2(i + 2)
            for i, d in enumerate(retrieved[:k])
            if d in relevant
        )
        ideal_hits = min(len(relevant), k)
        idcg = sum(1 / math.log2(i + 2) for i in range(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0


# ── Evaluator ──────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs retrieval for each evaluation query and computes IR metrics
    against MIRACL's official relevance judgments.
    """

    def __init__(self, engine, cross_lingual_retrieval=None):
        self.engine = engine
        self.clir   = cross_lingual_retrieval

    def run_evaluation(self, k: int = 10) -> dict:
        global EVALUATION_QUERIES
        if not EVALUATION_QUERIES:
            EVALUATION_QUERIES = load_queries()

        per_query = []
        for q in EVALUATION_QUERIES:
            lang       = q["language"]
            query_text = q["query"]
            relevant   = set(q.get("relevant_corpus_ids", []))

            # Retrieve using VSM for all languages
            results = self.engine.search(
                query_text,
                language=lang,
                top_k=k,
            )
            retrieved = [r["doc_id"] for r in results]

            p_k   = IRMetrics.precision_at_k(retrieved, relevant, k)
            r_k   = IRMetrics.recall_at_k(retrieved, relevant, k)
            f1    = IRMetrics.f1_at_k(retrieved, relevant, k)
            ap    = IRMetrics.average_precision(retrieved, relevant)
            ndcg  = IRMetrics.ndcg_at_k(retrieved, relevant, k)

            per_query.append({
                "query_id":  q["query_id"],
                "query":     query_text,
                "language":  lang,
                "category":  lang,   # use language as category for MIRACL
                f"p@{k}":    p_k,
                f"r@{k}":    r_k,
                f"f1@{k}":   f1,
                "ap":        ap,
                f"ndcg@{k}": ndcg,
                # alias keys used by app.py
                "precision": p_k,
                "recall":    r_k,
                "f1":        f1,
                "ndcg":      ndcg,
                "n_relevant": len(relevant),
            })

        map_score  = float(np.mean([q["ap"] for q in per_query]))
        avg_p      = float(np.mean([q[f"p@{k}"] for q in per_query]))
        avg_r      = float(np.mean([q[f"r@{k}"] for q in per_query]))
        avg_f1     = float(np.mean([q[f"f1@{k}"] for q in per_query]))
        avg_ndcg   = float(np.mean([q[f"ndcg@{k}"] for q in per_query]))

        return {
            "k":             k,
            "map":           map_score,
            "avg_precision": avg_p,
            "avg_recall":    avg_r,
            "avg_f1":        avg_f1,
            "avg_ndcg":      avg_ndcg,
            "queries":       per_query,
        }

    # ── Reporting ──────────────────────────────────────────────────────────────

    def print_report(self, results: dict) -> None:
        k = results["k"]
        print("\n" + "=" * 90)
        print("  IR Evaluation Report  (MIRACL EN/ES/FR/DE dev set)  (k={})".format(k))
        print("=" * 90)
        hdr = f"  {'QID':8s}  {'Lang':4s}  {'AP':6s}  {'P@k':6s}  {'R@k':6s}  {'F1@k':6s}  {'nDCG@k':7s}  Query"
        print(hdr)
        print("  " + "-" * 85)
        for q in results["queries"]:
            print(f"  {q['query_id']:8s}  {q['language']:4s}  "
                  f"{q['ap']:.3f}  {q[f'p@{k}']:.3f}  "
                  f"{q[f'r@{k}']:.3f}  {q[f'f1@{k}']:.3f}  "
                  f"{q[f'ndcg@{k}']:.3f}   {q['query'][:50]}")
        print("\n  " + "─" * 85)
        print(f"  AGGREGATE  MAP={results['map']:.4f}  "
              f"P@{k}={results['avg_precision']:.4f}  "
              f"R@{k}={results['avg_recall']:.4f}  "
              f"F1@{k}={results['avg_f1']:.4f}  "
              f"nDCG@{k}={results['avg_ndcg']:.4f}")
        print("=" * 90)

    def plot_precision_at_k(self, results: dict, max_k: int = 10) -> None:
        ks   = list(range(1, max_k + 1))
        langs = ["en", "es", "fr", "de"]
        colors = {"en": "#7f5af0", "es": "#e63946", "fr": "#2cb67d", "de": "#ff8e3c"}

        fig, ax = plt.subplots(figsize=(9, 5))
        for lang in langs:
            lang_queries = [q for q in results["queries"] if q["language"] == lang]
            if not lang_queries:
                continue
            pk_vals = []
            for ki in ks:
                avg = np.mean([
                    IRMetrics.precision_at_k(
                        self.engine.search(q["query"], language=lang, top_k=ki),
                        set(EVALUATION_QUERIES[[eq["query_id"] for eq in EVALUATION_QUERIES]
                                                .index(q["query_id"])
                                               if q["query_id"] in
                                               [eq["query_id"] for eq in EVALUATION_QUERIES]
                                               else 0
                                               ].get("relevant_corpus_ids", [])),
                        ki,
                    )
                    for q in lang_queries
                ])
                pk_vals.append(avg)
            ax.plot(ks, pk_vals, marker="o", label=lang.upper(), color=colors.get(lang))

        ax.set_xlabel("k")
        ax.set_ylabel("Precision@k")
        ax.set_title("Precision@k by Language — MIRACL Corpus")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    def plot_metrics_by_language(self, results: dict) -> None:
        langs = ["en", "es", "fr", "de"]
        k = results["k"]
        rows = []
        for lang in langs:
            qs = [q for q in results["queries"] if q["language"] == lang]
            if not qs:
                continue
            rows.append({
                "Language": lang.upper(),
                "MAP":      np.mean([q["ap"] for q in qs]),
                f"P@{k}":   np.mean([q[f"p@{k}"] for q in qs]),
                f"nDCG@{k}": np.mean([q[f"ndcg@{k}"] for q in qs]),
            })
        df = pd.DataFrame(rows).set_index("Language")
        df.plot(kind="bar", figsize=(8, 4), colormap="tab10")
        plt.title("Metrics by Language — MIRACL")
        plt.ylabel("Score")
        plt.xticks(rotation=0)
        plt.legend(loc="upper right")
        plt.tight_layout()


if __name__ == "__main__":
    print(f"Loaded {len(EVALUATION_QUERIES)} evaluation queries")
    for q in EVALUATION_QUERIES[:3]:
        print(f"  {q['query_id']} [{q['language']}]: {q['query']}")
        print(f"    relevant_corpus_ids: {q.get('relevant_corpus_ids', [])[:3]}")
