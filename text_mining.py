# text_mining.py
"""
Text Mining Component for the MIRACL multilingual IR system.

Techniques:
1. Document Clustering  — KMeans (k=12) on TF-IDF + LSA (TruncatedSVD)
2. Topic Modelling      — Latent Dirichlet Allocation (LDA, sklearn)
3. Query Expansion      — cluster top-terms injected into query
4. Keyphrase Extraction — per-document TF-IDF top-N keyphrases + query expansion

All classes expose a common ``expand_query(query, top_n)`` interface for use
by the search pipeline.

Author  : Thoshith S
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, PCA, LatentDirichletAllocation
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score


# ══════════════════════════════════════════════════════════════════════════════
# 1 · DOCUMENT CLUSTERING  (KMeans + LSA)
# ══════════════════════════════════════════════════════════════════════════════

class DocumentClusterer:
    """
    Cluster English documents using KMeans on a TF-IDF + LSA (TruncatedSVD)
    representation.  Cluster term profiles are used to expand queries with
    semantically related vocabulary, improving recall in downstream retrieval.
    """

    def __init__(self, corpus: list, n_clusters: int = 12, random_state: int = 42):
        self.corpus = corpus
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.vectorizer = None
        self.svd = None
        self.km = None
        self.lsa_matrix = None          # L2-normalised LSA vectors (N_en, n_components)
        self.cluster_labels = None      # (N_en,) int array
        self.en_doc_ids = []
        self.en_docs = []
        self.cluster_descriptions = {}  # cluster_id -> list[str] top terms
        self.silhouette = None

    # ------------------------------------------------------------------
    def fit(self):
        """TF-IDF → LSA → KMeans pipeline on English documents only."""
        self.en_docs   = [d for d in self.corpus if d.get('language') == 'en']
        self.en_doc_ids = [d['doc_id'] for d in self.en_docs]

        if not self.en_docs:
            raise ValueError("No English documents found in corpus.")

        texts = [f"{d['title']}. {d['text']}" for d in self.en_docs]

        self.vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            sublinear_tf=True, stop_words='english', min_df=2,
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        n_components = min(100, len(self.en_docs) - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        lsa_raw = self.svd.fit_transform(tfidf_matrix)
        self.lsa_matrix = normalize(lsa_raw, norm='l2')

        self.km = KMeans(
            n_clusters=self.n_clusters, n_init=10,
            random_state=self.random_state, init='k-means++',
        )
        self.cluster_labels = self.km.fit_predict(self.lsa_matrix)

        feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.cluster_descriptions = {}
        for cid in range(self.n_clusters):
            mask = self.cluster_labels == cid
            if not mask.any():
                self.cluster_descriptions[cid] = []
                continue
            mean_tfidf = np.asarray(tfidf_matrix[mask].mean(axis=0)).flatten()
            top_idx = mean_tfidf.argsort()[-5:][::-1]
            self.cluster_descriptions[cid] = list(feature_names[top_idx])

        sample = min(1000, len(self.lsa_matrix))
        try:
            self.silhouette = silhouette_score(
                self.lsa_matrix, self.cluster_labels,
                metric='cosine', sample_size=sample, random_state=self.random_state,
            )
        except Exception:
            self.silhouette = float('nan')

        self._print_cluster_summary()

    # ------------------------------------------------------------------
    def _print_cluster_summary(self):
        print(f"\nDocument Clustering Summary")
        print(f"  English docs  : {len(self.en_docs)}")
        print(f"  Clusters (k)  : {self.n_clusters}")
        print(f"  Silhouette    : {self.silhouette:.4f}")
        print(f"  {'Cluster':>7}  {'Size':>5}  Top Terms")
        print(f"  {'-'*65}")
        for cid in range(self.n_clusters):
            size  = int((self.cluster_labels == cid).sum())
            terms = ', '.join(self.cluster_descriptions.get(cid, []))
            print(f"  {cid:>7}  {size:>5}  {terms}")

    # ------------------------------------------------------------------
    def get_cluster_for_query(self, query: str) -> int:
        if self.vectorizer is None:
            raise RuntimeError("Call fit() first.")
        tfidf_q = self.vectorizer.transform([query])
        lsa_q   = normalize(self.svd.transform(tfidf_q), norm='l2')
        centroids_normed = normalize(self.km.cluster_centers_, norm='l2')
        sims = (lsa_q @ centroids_normed.T).flatten()
        return int(np.argmax(sims))

    def expand_query(self, query: str, top_n: int = 5) -> list:
        """Return cluster-derived expansion terms not in the query."""
        cid = self.get_cluster_for_query(query)
        qtoks = set(query.lower().split())
        return [t for t in self.cluster_descriptions.get(cid, []) if t not in qtoks][:top_n]

    def get_cluster_documents(self, cluster_id: int) -> list:
        if self.cluster_labels is None:
            raise RuntimeError("Call fit() first.")
        return [self.en_docs[i] for i, lbl in enumerate(self.cluster_labels) if lbl == cluster_id]

    def get_cluster_summary(self) -> pd.DataFrame:
        if self.cluster_labels is None:
            raise RuntimeError("Call fit() first.")
        rows = []
        for cid in range(self.n_clusters):
            mask = self.cluster_labels == cid
            docs = [self.en_docs[i] for i in np.where(mask)[0]]
            rows.append({
                'cluster_id': cid,
                'size': int(mask.sum()),
                'top_terms': ', '.join(self.cluster_descriptions.get(cid, [])),
                'sample_titles': ', '.join(d['title'] for d in docs[:3]),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------  Plots
    def plot_clusters(self, save_path: str = None):
        if self.lsa_matrix is None:
            raise RuntimeError("Call fit() first.")
        pca    = PCA(n_components=2, random_state=self.random_state)
        coords = pca.fit_transform(self.lsa_matrix)
        fig, ax = plt.subplots(figsize=(12, 8))
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=self.cluster_labels, cmap='tab20', s=6, alpha=0.7)
        plt.colorbar(sc, ax=ax, label='Cluster')
        ax.set_title("Document Clusters — MIRACL EN Corpus")
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150); print(f"Saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def plot_cluster_sizes(self, save_path: str = None):
        if self.cluster_labels is None:
            raise RuntimeError("Call fit() first.")
        sizes  = [int((self.cluster_labels == c).sum()) for c in range(self.n_clusters)]
        labels = [f"C{c}" for c in range(self.n_clusters)]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(np.arange(len(labels)), sizes, align='center')
        ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
        ax.invert_yaxis(); ax.set_xlabel("Documents"); ax.set_title("Cluster Size Distribution")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        else:
            plt.show()
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 2 · TOPIC MODELLING  (LDA)
# ══════════════════════════════════════════════════════════════════════════════

class TopicModeller:
    """
    Latent Dirichlet Allocation (LDA) topic modelling on English documents.

    Discovers latent topics as probability distributions over vocabulary terms.
    Each query is projected into the topic space; the dominant topic's top words
    are appended to the query for semantic expansion — capturing thematic context
    that exact-match TF-IDF misses.
    """

    def __init__(self, corpus: list, n_topics: int = 10, random_state: int = 42):
        self.corpus       = corpus
        self.n_topics     = n_topics
        self.random_state = random_state

        self.vectorizer        = None     # CountVectorizer (raw TF)
        self.lda               = None     # LatentDirichletAllocation
        self.doc_topic_matrix  = None     # (N_en, n_topics)
        self.topic_words       = {}       # topic_id → list[str]
        self.en_doc_ids        = []
        self.en_docs           = []
        self._is_fitted        = False

    # ------------------------------------------------------------------
    def fit(self):
        """Fit LDA on English documents using raw term counts."""
        self.en_docs    = [d for d in self.corpus if d.get('language') == 'en']
        self.en_doc_ids = [d['doc_id'] for d in self.en_docs]

        if not self.en_docs:
            raise ValueError("No English documents found.")

        texts = [f"{d['title']}. {d['text']}" for d in self.en_docs]

        # CountVectorizer — LDA assumes multinomial (count) input
        self.vectorizer = CountVectorizer(
            max_features=3000,
            stop_words='english',
            min_df=2,
            ngram_range=(1, 1),
        )
        count_matrix = self.vectorizer.fit_transform(texts)

        self.lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            max_iter=30,
            learning_method='online',
            learning_offset=10.0,
        )
        self.doc_topic_matrix = self.lda.fit_transform(count_matrix)  # (N, n_topics)

        feature_names = np.array(self.vectorizer.get_feature_names_out())
        for tid, topic_dist in enumerate(self.lda.components_):
            top_idx = topic_dist.argsort()[-12:][::-1]
            self.topic_words[tid] = list(feature_names[top_idx])

        self._is_fitted = True
        self._print_topics()

    # ------------------------------------------------------------------
    def _print_topics(self):
        print(f"\nLDA Topic Modelling Summary")
        print(f"  English docs : {len(self.en_docs)}")
        print(f"  Topics (k)   : {self.n_topics}")
        for tid, words in self.topic_words.items():
            print(f"  Topic {tid:2d}: {', '.join(words[:6])}")

    # ------------------------------------------------------------------
    def get_dominant_topic(self, query: str) -> int:
        """Return the dominant topic index for a query."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        q_vec      = self.vectorizer.transform([query])
        topic_dist = self.lda.transform(q_vec)[0]
        return int(np.argmax(topic_dist))

    def get_topic_distribution(self, query: str) -> np.ndarray:
        """Return full topic probability vector for a query (sums to 1)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        return self.lda.transform(self.vectorizer.transform([query]))[0]

    def expand_query(self, query: str, top_n: int = 5) -> list:
        """Return LDA topic words not already present in the query."""
        tid   = self.get_dominant_topic(query)
        qtoks = set(query.lower().split())
        return [w for w in self.topic_words.get(tid, []) if w not in qtoks][:top_n]

    # ------------------------------------------------------------------
    def get_topic_summary(self) -> pd.DataFrame:
        """DataFrame: topic_id, perplexity-weighted top words, representative docs."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        rows = []
        for tid, words in self.topic_words.items():
            top_doc_idx = np.argsort(self.doc_topic_matrix[:, tid])[-3:][::-1]
            sample_docs = ', '.join(self.en_docs[i]['title'] for i in top_doc_idx)
            rows.append({
                'topic_id':   tid,
                'top_words':  ', '.join(words[:8]),
                'sample_docs': sample_docs,
            })
        return pd.DataFrame(rows)

    def plot_topic_heatmap(self, save_path: str = None):
        """Heatmap of per-document dominant topic assignments."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        dominant = np.argmax(self.doc_topic_matrix, axis=1)
        counts   = np.bincount(dominant, minlength=self.n_topics)

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(
            self.doc_topic_matrix.T,
            aspect='auto', cmap='YlOrRd', interpolation='nearest',
        )
        plt.colorbar(im, ax=ax, label='Topic probability')
        ax.set_xlabel("Document index"); ax.set_ylabel("Topic ID")
        ax.set_title("LDA Document-Topic Matrix (English corpus)")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        else:
            plt.show()
        plt.close(fig)
        return counts


# ══════════════════════════════════════════════════════════════════════════════
# 3 · KEYPHRASE EXTRACTION  (TF-IDF per document)
# ══════════════════════════════════════════════════════════════════════════════

class KeyphraseExtractor:
    """
    TF-IDF based keyphrase extraction.

    Builds a corpus-wide TF-IDF model.  For each document the highest-weighted
    terms are its keyphrases.  For query expansion, the query is projected into
    the same TF-IDF space and its top terms (not already in the query) are
    returned — focusing retrieval on the most discriminative vocabulary.
    """

    def __init__(self, corpus: list):
        self.corpus          = corpus
        self.vectorizer      = None     # TfidfVectorizer (whole corpus)
        self.tfidf_matrix    = None     # sparse (N_docs, vocab)
        self.doc_ids         = []
        self.doc_id_to_idx   = {}
        self._is_fitted      = False

    # ------------------------------------------------------------------
    def fit(self):
        """Fit TF-IDF vectorizer on the full multilingual corpus."""
        texts          = [f"{d['title']}. {d['text']}" for d in self.corpus]
        self.doc_ids   = [d['doc_id'] for d in self.corpus]
        self.doc_id_to_idx = {did: i for i, did in enumerate(self.doc_ids)}

        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self._is_fitted   = True
        print(f"KeyphraseExtractor fitted on {len(self.corpus)} documents  "
              f"(vocab {self.tfidf_matrix.shape[1]:,})")

    # ------------------------------------------------------------------
    def extract_keyphrases(self, doc_id: str, n: int = 5) -> list:
        """Return top-n keyphrases for a single document."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        idx = self.doc_id_to_idx.get(doc_id)
        if idx is None:
            return []
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        row           = np.asarray(self.tfidf_matrix[idx].todense()).flatten()
        top_idx       = row.argsort()[-n:][::-1]
        return [feature_names[i] for i in top_idx if row[i] > 0]

    def expand_query(self, query: str, top_n: int = 5) -> list:
        """
        Transform the query string through the corpus TF-IDF model and return
        top-n high-IDF terms not already in the query as expansion tokens.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        q_vec         = self.vectorizer.transform([query])
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        row           = np.asarray(q_vec.todense()).flatten()
        top_idx       = row.argsort()[-top_n * 3:][::-1]
        qtoks         = set(query.lower().split())
        result        = []
        for i in top_idx:
            if row[i] > 0:
                term = feature_names[i]
                if term not in qtoks:
                    result.append(term)
            if len(result) >= top_n:
                break
        return result

    # ------------------------------------------------------------------
    def get_keyphrases_batch(self, n: int = 5) -> pd.DataFrame:
        """Return a DataFrame of doc_id, title, language, keyphrases for all docs."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        rows = []
        for doc in self.corpus:
            kp = self.extract_keyphrases(doc['doc_id'], n=n)
            rows.append({
                'doc_id':     doc['doc_id'],
                'title':      doc['title'],
                'language':   doc.get('language', ''),
                'keyphrases': ', '.join(kp),
            })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4 · TEXT MINING COMPARATOR  (benchmark all expansion strategies)
# ══════════════════════════════════════════════════════════════════════════════

def compare_text_mining_methods(
    engine,
    clusterer: DocumentClusterer,
    topic_modeller: TopicModeller,
    kp_extractor: KeyphraseExtractor,
    queries: list,
    k: int = 10,
) -> pd.DataFrame:
    """
    Benchmark four text mining strategies against the official MIRACL qrels.

    For each strategy the engine is asked to retrieve top-k documents for every
    evaluation query.  Average Precision (AP) and nDCG@k are computed, then
    averaged across queries → MAP and mean nDCG.

    Parameters
    ----------
    engine         : fitted SearchEngine instance
    clusterer      : fitted DocumentClusterer
    topic_modeller : fitted TopicModeller
    kp_extractor   : fitted KeyphraseExtractor
    queries        : list of eval query dicts (with 'relevant_corpus_ids')
    k              : rank cutoff

    Returns
    -------
    DataFrame with columns: Method, MAP, nDCG@k, Δ_MAP (vs baseline)
    """
    methods = {
        'Baseline (no expansion)':   None,
        'Clustering (KMeans+LSA)':   clusterer,
        'Topic Modelling (LDA)':     topic_modeller,
        'Keyphrase Extraction (TF-IDF)': kp_extractor,
    }

    summary = {}

    for method_name, expander in methods.items():
        ap_list   = []
        ndcg_list = []

        for q in queries:
            relevant = set(q.get('relevant_corpus_ids', []))
            if not relevant:
                continue

            lang       = q.get('language', 'en')
            query_text = q['query']

            exp_terms = []
            if expander is not None:
                try:
                    exp_terms = expander.expand_query(query_text, top_n=5)
                except Exception:
                    exp_terms = []

            res_list = engine.search(
                query_text,
                language=lang,
                top_k=k,
                expanded_terms=exp_terms or None,
            )
            retrieved = [r['doc_id'] for r in res_list]

            # AP
            hits = 0
            prec_sum = 0.0
            for rank, did in enumerate(retrieved, 1):
                if did in relevant:
                    hits += 1
                    prec_sum += hits / rank
            ap = prec_sum / len(relevant)
            ap_list.append(ap)

            # nDCG@k
            dcg      = sum(1.0 / np.log2(r + 1)
                           for r, did in enumerate(retrieved, 1) if did in relevant)
            ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
            ndcg      = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            ndcg_list.append(ndcg)

        summary[method_name] = {
            'MAP':        float(np.mean(ap_list))   if ap_list   else 0.0,
            f'nDCG@{k}': float(np.mean(ndcg_list)) if ndcg_list else 0.0,
        }

    # Build DataFrame with Δ MAP relative to baseline
    baseline_map = summary['Baseline (no expansion)']['MAP']
    rows = []
    for method, metrics in summary.items():
        delta = metrics['MAP'] - baseline_map
        rows.append({
            'Method':     method,
            'MAP':        round(metrics['MAP'],        4),
            f'nDCG@{k}': round(metrics[f'nDCG@{k}'], 4),
            'Δ MAP':      round(delta,                 4),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Legacy helper (kept for backwards compat)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_query_expansion(search_engine, clusterer, test_queries: list) -> pd.DataFrame:
    """
    Compare mean top-5 retrieval score with / without cluster-based expansion.
    Returns DataFrame: query, original_score, expanded_score, expansion_terms, improvement_%
    """
    rows = []
    for query in test_queries:
        orig   = search_engine.search(query, top_k=5)
        o_sc   = float(np.mean([r['score'] for r in orig])) if orig else 0.0

        exp_terms    = clusterer.expand_query(query, top_n=5)
        exp_query    = query + ' ' + ' '.join(exp_terms) if exp_terms else query
        expanded     = search_engine.search(exp_query, top_k=5)
        e_sc         = float(np.mean([r['score'] for r in expanded])) if expanded else 0.0

        impr = 100.0 * (e_sc - o_sc) / o_sc if o_sc > 0 else 0.0
        rows.append({
            'query':           query,
            'original_score':  round(o_sc,   4),
            'expanded_score':  round(e_sc,   4),
            'expansion_terms': ', '.join(exp_terms),
            'improvement_%':   round(impr,   2),
        })
    return pd.DataFrame(rows)
