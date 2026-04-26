"""
search_engine.py
Vector Space Model with TF-IDF weighting and cosine similarity.
Provides ranked top-k retrieval with sentence-aware snippet generation and
query term highlighting for the MIRACL multilingual corpus (EN, ES, FR, DE).

Author  : Thoshith S
"""

import math
import re
from collections import defaultdict
from preprocessor import tokenize_for_lang, tokenize_english


class SearchEngine:
    """
    TF-IDF Vector Space Model search engine.

    Documents are represented as TF-IDF weighted vectors.
    Retrieval is by cosine similarity between the query vector and document vectors.
    """

    def __init__(self,
                 corpus: list[dict],
                 preprocessed_corpus: list[dict],
                 inverted_index: dict,
                 doc_lengths: dict):
        self.corpus              = corpus
        self.preprocessed_corpus = preprocessed_corpus
        self.inverted_index      = inverted_index
        self.doc_lengths         = doc_lengths
        self.N                   = len(corpus)

        # Fast-lookup dicts
        self.doc_id_to_doc  = {d["doc_id"]: d for d in corpus}
        self.doc_id_to_prep = {d["doc_id"]: d for d in preprocessed_corpus}

        # Build TF-IDF representation
        self._compute_tfidf()
        print(f"SearchEngine ready  |  {self.N} docs  |  vocab {len(self.idf):,}")

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    def _compute_tfidf(self) -> None:
        """Compute IDF, TF-IDF vectors, and document L2 norms."""
        # IDF: log(N / df)  with df = number of docs containing term
        self.idf: dict[str, float] = {}
        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.idf[term] = math.log(self.N / df) if df > 0 else 0.0

        # TF-IDF vectors per doc:  {doc_id: {term: weight}}
        # TF = raw_count / doc_length  (normalized TF)
        self.tfidf: dict[str, dict[str, float]] = defaultdict(dict)
        for term, postings in self.inverted_index.items():
            idf = self.idf[term]
            for doc_id, raw_tf in postings:
                dlen = self.doc_lengths.get(doc_id, 1)
                tf   = raw_tf / dlen
                self.tfidf[doc_id][term] = tf * idf

        # Pre-compute L2 norms for each document vector
        self.doc_norms: dict[str, float] = {}
        for doc_id, vec in self.tfidf.items():
            self.doc_norms[doc_id] = math.sqrt(sum(v * v for v in vec.values()))

    # ── Query preprocessing ────────────────────────────────────────────────────
    def _preprocess_query(self, query: str, language: str = "en") -> list[str]:
        return tokenize_for_lang(query, language)

    # ── Cosine similarity ──────────────────────────────────────────────────────
    def _cosine_similarity(self, query_vec: dict[str, float], doc_id: str) -> float:
        doc_vec  = self.tfidf.get(doc_id, {})
        dot      = sum(query_vec.get(t, 0.0) * doc_vec.get(t, 0.0)
                       for t in query_vec)
        q_norm   = math.sqrt(sum(v * v for v in query_vec.values()))
        d_norm   = self.doc_norms.get(doc_id, 0.0)
        if q_norm == 0 or d_norm == 0:
            return 0.0
        return dot / (q_norm * d_norm)

    # ── Search ─────────────────────────────────────────────────────────────────
    def search(self,
               query: str,
               language: str = "en",
               top_k: int = 10,
               target_lang: str = None,
               expanded_terms: list[str] = None) -> list[dict]:
        """
        Retrieve top-k documents using TF-IDF cosine similarity.

        Args:
            query        : raw query string
            language     : query language code ('en', 'es', 'fr', or 'de')
            top_k        : number of results to return
            target_lang  : restrict results to this language (or None for all languages)
            expanded_terms: extra tokens from query expansion (text mining)

        Returns:
            list of result dicts with rank, doc_id, title, score, snippet, language, url
        """
        if not query.strip():
            return []

        tokens = self._preprocess_query(query, language)
        if expanded_terms:
            tokens = tokens + expanded_terms

        if not tokens:
            return []

        # Build query TF-IDF vector
        token_counts = defaultdict(int)
        for t in tokens:
            token_counts[t] += 1

        query_vec: dict[str, float] = {}
        for term, cnt in token_counts.items():
            if term in self.idf:
                tf = cnt / len(tokens)
                query_vec[term] = tf * self.idf[term]

        if not query_vec:
            return []

        # Collect candidate docs (only docs sharing at least one query term)
        candidates: set[str] = set()
        for term in query_vec:
            for doc_id, _ in self.inverted_index.get(term, []):
                if target_lang is None:
                    candidates.add(doc_id)
                else:
                    doc = self.doc_id_to_doc.get(doc_id, {})
                    if doc.get("language") == target_lang:
                        candidates.add(doc_id)

        if not candidates:
            return []

        # Score and rank
        scored = []
        for doc_id in candidates:
            score = self._cosine_similarity(query_vec, doc_id)
            if score > 0:
                scored.append((doc_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        results = []
        for rank, (doc_id, score) in enumerate(top, 1):
            doc  = self.doc_id_to_doc.get(doc_id, {})
            _    = self.doc_id_to_prep.get(doc_id, {})
            snippet     = self._generate_snippet(doc, tokens)
            highlighted = self._highlight_terms(snippet, tokens, original_query=query)
            hl_title    = self._highlight_title(doc.get("title", ""), tokens, original_query=query)
            results.append({
                "rank":               rank,
                "doc_id":             doc_id,
                "title":              doc.get("title", ""),
                "highlighted_title":  hl_title,
                "score":              round(score, 6),
                "snippet":            snippet,
                "highlighted_snippet": highlighted,
                "language":           doc.get("language", ""),
                "url":                doc.get("url", ""),
            })

        return results

    # ── Snippet generation ─────────────────────────────────────────────────────
    def _generate_snippet(self, doc: dict, query_tokens: list[str],
                          max_chars: int = 300) -> str:
        """
        Extract 1-3 most relevant sentences containing query terms (sentence-aware).
        Returns sentences joined with ' … ' separators — Google-style snippet.
        """
        text = doc.get("text", "")
        if not text:
            return ""

        # Split into sentences at punctuation boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return text[:max_chars] + ("…" if len(text) > max_chars else "")

        token_set = set(query_tokens)

        # Score each sentence: count of distinct query tokens that appear in it
        scored = [
            (sum(1 for t in token_set if t in s.lower()), i, s)
            for i, s in enumerate(sentences)
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))

        # Pick up to 3 top sentences while staying within max_chars
        picked = []
        total  = 0
        for _hits, pos, sent in scored:
            if total + len(sent) > max_chars and picked:
                break
            picked.append((pos, sent))
            total += len(sent)
            if len(picked) >= 3:
                break

        # Restore reading order
        picked.sort(key=lambda x: x[0])
        return " … ".join(s for _, s in picked) or text[:max_chars]

    # ── Term highlighting ──────────────────────────────────────────────────────
    def _highlight_terms(self, snippet: str, query_tokens: list[str],
                         original_query: str = "") -> str:
        """
        HTML-escape the snippet, then wrap matching terms in <mark> tags.
        Highlights both original query words and processed/stemmed tokens.
        """
        import html as _html
        if not snippet:
            return ""
        text = _html.escape(snippet)

        # Collect terms: original surface words + stemmed tokens
        terms: set[str] = set(query_tokens)
        if original_query:
            terms.update(w for w in re.findall(r'\b\w{3,}\b', original_query))

        for term in sorted(terms, key=len, reverse=True):
            if len(term) < 3:
                continue
            pattern = re.compile(
                r'(?<![a-zA-Z\u00C0-\u017E])' + re.escape(term) +
                r'(?![a-zA-Z\u00C0-\u017E])',
                re.IGNORECASE,
            )
            text = pattern.sub(
                lambda m: f'<mark class="qterm">{m.group()}</mark>', text
            )
        return text

    def _highlight_title(self, title: str, query_tokens: list[str],
                         original_query: str = "") -> str:
        """Return HTML-escaped title with query terms highlighted."""
        return self._highlight_terms(title, query_tokens, original_query)

    # ── Document access ────────────────────────────────────────────────────────
    def get_document(self, doc_id: str) -> dict | None:
        return self.doc_id_to_doc.get(doc_id)

    # ── Statistics ─────────────────────────────────────────────────────────────
    def get_statistics(self) -> dict:
        by_lang: dict[str, int] = {}
        for d in self.corpus:
            lang = d.get("language", "?")
            by_lang[lang] = by_lang.get(lang, 0) + 1
        avg_len = sum(self.doc_lengths.values()) / max(len(self.doc_lengths), 1)
        return {
            "vocab_size":     len(self.idf),
            "n_docs":         self.N,
            "by_lang":        by_lang,
            "n_en":           by_lang.get("en", 0),
            "avg_doc_length": round(avg_len, 1),
            "index_size":     sum(len(v) for v in self.inverted_index.values()),
        }


# ── Formatting helper ──────────────────────────────────────────────────────────
def format_results(results: list[dict], show_snippets: bool = True) -> str:
    """Format search results as a human-readable string."""
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lang_badge = "🇬🇧 EN" if r["language"] == "en" else "🇮🇳 HI"
        lines.append(f"\n#{r['rank']}  [{lang_badge}]  {r['title']}")
        lines.append(f"    Score : {r['score']:.4f}")
        if show_snippets and r.get("highlighted_snippet"):
            lines.append(f"    ...{r['highlighted_snippet']}...")
        if r.get("url"):
            lines.append(f"    URL   : {r['url']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("SearchEngine module loaded OK")
