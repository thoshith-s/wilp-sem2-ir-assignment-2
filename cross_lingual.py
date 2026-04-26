# cross_lingual.py
"""
Cross-Lingual Retrieval via Shared Multilingual Embeddings (LaBSE).

Encodes corpus documents (EN, ES, FR, DE) into a shared 768-dim embedding
space using LaBSE (Language-agnostic BERT Sentence Embeddings).
A query in any language retrieves documents across all languages via cosine
similarity — no translation step required.

Author  : Thoshith S
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import html as _html
import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def _clir_snippet(doc: dict, query_words: list[str], max_chars: int = 300) -> tuple[str, str]:
    """
    Generate a sentence-aware snippet for a CLIR result and highlight query words.

    Returns (plain_snippet, highlighted_snippet_html).
    """
    text = doc.get("text", "")
    if not text:
        return "", ""

    # Sentence-aware extraction
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        plain = text[:max_chars] + ("…" if len(text) > max_chars else "")
        return plain, _html.escape(plain)

    token_set = set(w.lower() for w in query_words if len(w) >= 3)
    scored = [
        (sum(1 for t in token_set if t in s.lower()), i, s)
        for i, s in enumerate(sentences)
    ]
    scored.sort(key=lambda x: (-x[0], x[1]))

    picked, total = [], 0
    for _h, pos, sent in scored:
        if total + len(sent) > max_chars and picked:
            break
        picked.append((pos, sent))
        total += len(sent)
        if len(picked) >= 3:
            break

    picked.sort(key=lambda x: x[0])
    plain = " … ".join(s for _, s in picked) or text[:max_chars]

    # Highlight
    escaped = _html.escape(plain)
    for term in sorted(token_set, key=len, reverse=True):
        if len(term) < 3:
            continue
        pattern = re.compile(
            r'(?<![a-zA-Z\u00C0-\u017E])' + re.escape(term) +
            r'(?![a-zA-Z\u00C0-\u017E])',
            re.IGNORECASE,
        )
        escaped = pattern.sub(lambda m: f'<mark class="qterm">{m.group()}</mark>', escaped)

    return plain, escaped


def detect_language(text: str) -> str:
    """
    Detect language of text for the MIRACL EN/ES/FR/DE corpus.

    Strategy:
    1. Character-set heuristics (fast, reliable for distinctive chars)
    2. langdetect library (handles ambiguous Latin text)
    3. Fallback to 'en'
    """
    t = text.strip()
    if not t:
        return "en"

    # Spanish: inverted punctuation
    if "¿" in t or "¡" in t:
        return "es"

    # German: umlauts / Eszett
    if any(c in t for c in "äöüÄÖÜß"):
        return "de"

    # French: cedilla, œ, or specific accents
    if any(c in t for c in "êœâûîç"):
        return "fr"

    t_lower = t.lower()
    if any(m in t_lower for m in ("est-ce", "qu'est", "c'est", "n'est", "l'a", "d'un")):
        return "fr"

    _fr_words = {"quelles", "quel", "quelle", "sont", "dans", "avec", "sur",
                 "nous", "vous", "ils", "elles", "leur", "leurs"}
    _de_words = {"was", "ist", "die", "der", "das", "wie", "und", "ein", "eine",
                 "nicht", "auch", "mit", "bei", "von", "für", "welche", "welcher",
                 "welches", "beim", "einer", "einem"}
    _es_words = {"qué", "cuál", "cual", "cómo", "dónde", "cuándo", "cuánto",
                 "por", "los", "las", "del", "también", "está", "están"}

    words = set(t_lower.split())
    if words & _fr_words:
        return "fr"
    if words & _de_words:
        return "de"
    if words & _es_words:
        return "es"

    try:
        from langdetect import detect as _ld
        lang = _ld(t)
        if lang in ("en", "es", "fr", "de"):
            return lang
    except Exception:
        pass

    return "en"


def demo_cross_lingual_pairs() -> list:
    """Return cross-lingual query pairs for demo (MIRACL domain)."""
    return [
        ("What is anarchism?", "¿Qué es el anarquismo?"),
        ("History of the French Revolution", "Geschichte der Französischen Revolution"),
        ("Philosophy of science", "Philosophie des sciences"),
        ("Democracy and human rights", "Democracia y derechos humanos"),
        ("Climate change effects", "Effets du changement climatique"),
    ]


class MultilingualEmbeddingRetrieval:
    """
    Cross-lingual retrieval using LaBSE shared multilingual embedding space.
    Encodes all corpus documents (EN, ES, FR, DE) into a unified 768-dim space.
    Queries in any language can retrieve documents in any language via cosine
    similarity — no translation step needed.
    """

    def __init__(self, corpus: list, model_name: str = 'sentence-transformers/LaBSE'):
        self.corpus = corpus
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.embeddings = None          # shape (N_docs, 768)
        self.doc_ids = []
        self.doc_id_to_doc = {doc['doc_id']: doc for doc in corpus}
        self._is_fitted = False
        self._device = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load LaBSE tokenizer and model onto best available device."""
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self._device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)
        self.model.eval()
        print(f"LaBSE loaded on {device}")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts using LaBSE in batches.
        Returns L2-normalised float32 ndarray of shape (len(texts), 768).
        """
        if self.model is None or self.tokenizer is None:
            self._load_model()

        all_embeddings = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    max_length=128,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                output = self.model(**encoded)
                pooled = output.pooler_output           # (B, 768)
                normed = F.normalize(pooled, p=2, dim=1)
                all_embeddings.append(normed.cpu().float().numpy())

                if (i // batch_size) % 5 == 0 and i > 0:
                    print(f"  Encoded {i}/{len(texts)}...")

        return np.vstack(all_embeddings).astype(np.float32)

    # ------------------------------------------------------------------
    # Fitting (indexing)
    # ------------------------------------------------------------------

    def fit(self):
        """Encode all corpus documents and build the embedding index."""
        if self.model is None:
            self._load_model()

        # title + first 500 chars of body
        texts = [
            f"{doc['title']}. {doc['text'][:500]}"
            for doc in self.corpus
        ]

        print(f"Encoding {len(texts)} documents...")
        self.embeddings = self._encode(texts)       # (N, 768), already L2-normed
        self.doc_ids = [doc['doc_id'] for doc in self.corpus]

        # Free device memory
        if self._device == 'mps':
            torch.mps.empty_cache()

        by_lang = {}
        for d in self.corpus:
            by_lang.setdefault(d.get('language', '?'), 0)
            by_lang[d.get('language', '?')] += 1
        self._is_fitted = True
        lang_str = "  ".join(f"{k.upper()}={v}" for k, v in sorted(by_lang.items()))
        print(f"Indexed {len(self.doc_ids)} documents  ({lang_str})")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, target_lang: str = None, top_k: int = 10) -> list:
        """
        Retrieve top_k documents most similar to the query.

        Parameters
        ----------
        query       : query string (any language)
        target_lang : if set, restrict results to that language
        top_k       : number of results to return
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() or load_embeddings() before search().")

        query_vec = self._encode([query])                       # (1, 768)
        scores    = (self.embeddings @ query_vec.T).flatten()   # cosine sim

        # Optional language filter
        if target_lang is not None:
            for idx, doc_id in enumerate(self.doc_ids):
                if self.doc_id_to_doc[doc_id].get('language') != target_lang:
                    scores[idx] = -np.inf

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        # Query words for snippet highlighting (surface-form, no stemming needed for CLIR)
        query_words = re.findall(r'\b\w{3,}\b', query)

        results = []
        for rank, idx in enumerate(ranked_indices):
            if scores[idx] == -np.inf:
                break
            doc_id = self.doc_ids[idx]
            doc    = self.doc_id_to_doc[doc_id]
            plain_snippet, hl_snippet = _clir_snippet(doc, query_words)
            hl_title = _html.escape(doc.get('title', ''))
            # Highlight title too
            for term in sorted(set(w.lower() for w in query_words if len(w) >= 3),
                               key=len, reverse=True):
                pattern = re.compile(
                    r'(?<![a-zA-Z\u00C0-\u017E])' + re.escape(term) +
                    r'(?![a-zA-Z\u00C0-\u017E])', re.IGNORECASE)
                hl_title = pattern.sub(
                    lambda m: f'<mark class="qterm">{m.group()}</mark>', hl_title)
            results.append({
                "rank":                rank + 1,
                "doc_id":              doc_id,
                "title":               doc.get('title', ''),
                "highlighted_title":   hl_title,
                "score":               float(scores[idx]),
                "snippet":             plain_snippet,
                "highlighted_snippet": hl_snippet,
                "language":            doc.get('language', ''),
                "url":                 doc.get('url', ''),
            })

        return results

    # ------------------------------------------------------------------
    # Cross-lingual search
    # ------------------------------------------------------------------

    def search_cross_lingual(self, query: str, source_lang: str = None,
                             top_k: int = 10) -> dict:
        """
        Full cross-lingual search: retrieve from all languages, then
        separately from EN-only and HI-only pools.
        """
        if source_lang is None:
            source_lang = detect_language(query)

        multilingual_results = self.search(query, target_lang=None, top_k=top_k)
        lang_results = {}
        for lang in ["en", "es", "fr", "de"]:
            lang_results[f"{lang}_results"] = self.search(query, target_lang=lang, top_k=top_k)

        return {
            "query": query,
            "detected_language": source_lang,
            "multilingual_results": multilingual_results,
            "model": self.model_name,
            **lang_results,
        }

    # ------------------------------------------------------------------
    # Cross-lingual similarity between two queries
    # ------------------------------------------------------------------

    def compute_cross_lingual_similarity(self, query_a: str, query_b: str) -> float:
        """
        Compute cosine similarity between two queries in the shared LaBSE space.
        Useful to verify cross-lingual semantic equivalence (e.g. EN ↔ ES ↔ FR ↔ DE).
        """
        vecs = self._encode([query_a, query_b])   # (2, 768)
        similarity = float(np.dot(vecs[0], vecs[1]))
        return similarity

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_embeddings(self, path: str = 'doc_embeddings.npz'):
        """Persist document embeddings and doc_ids to a .npz file."""
        np.savez(path, embeddings=self.embeddings, doc_ids=np.array(self.doc_ids))
        print(f"Embeddings saved to {path}")

    def load_embeddings(self, path: str = 'doc_embeddings.npz'):
        """Load document embeddings and doc_ids from a .npz file."""
        data = np.load(path, allow_pickle=True)
        self.embeddings = data['embeddings'].astype(np.float32)
        self.doc_ids = list(data['doc_ids'])
        self._is_fitted = True
        print(f"Loaded embeddings {self.embeddings.shape} from {path}")

    # ------------------------------------------------------------------
    # Document similarity
    # ------------------------------------------------------------------

    def get_similar_documents(self, doc_id: str, top_k: int = 5) -> list:
        """
        Find the most similar documents to the given document.
        Useful for cross-lingual document similarity
        (e.g. EN "Shah Rukh Khan" article ↔ HI "शाहरुख खान" article).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() or load_embeddings() before get_similar_documents().")

        try:
            idx = self.doc_ids.index(doc_id)
        except ValueError:
            raise ValueError(f"doc_id '{doc_id}' not found in index.")

        query_vec = self.embeddings[idx]                        # (768,)
        scores = (self.embeddings @ query_vec).flatten()        # cosine sim
        scores[idx] = -np.inf                                   # exclude self

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, i in enumerate(ranked_indices):
            d_id = self.doc_ids[i]
            doc = self.doc_id_to_doc[d_id]
            plain, _ = _clir_snippet(doc, [])
            results.append({
                "rank":    rank + 1,
                "doc_id":  d_id,
                "title":   doc.get('title', ''),
                "score":   float(scores[i]),
                "snippet": plain,
                "language": doc.get('language', ''),
                "url":     doc.get('url', ''),
            })

        return results
