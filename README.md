# MIRACL Multilingual IR Search Engine

A multilingual information retrieval system over the [MIRACL](https://huggingface.co/datasets/miracl/miracl-corpus) Wikipedia corpus covering English, Spanish, French, and German.

**Author:** Thoshith S  
**Course:** BITS Pilani WILP — AIMLZG537 Information Retrieval — Assignment 2 · Group 16  
**Live Demo:** [huggingface.co/spaces/thoshiths/miracl-search](https://huggingface.co/spaces/thoshiths/miracl-search)  
**Dataset:** [thoshiths/miracl-multilingual-4M](https://huggingface.co/datasets/thoshiths/miracl-multilingual-4M)

---

## Overview

This project implements a complete multilingual IR pipeline with two retrieval methods, four text mining techniques, and a Streamlit web interface deployed on HuggingFace Spaces. The system supports queries in English, Spanish, French, and German and retrieves relevant Wikipedia passages across all four languages.

---

## Corpus

**Source:** [`thoshiths/miracl-multilingual-4M`](https://huggingface.co/datasets/thoshiths/miracl-multilingual-4M)  
**Scale:** 400,000 Wikipedia passages — 100,000 per language (EN · ES · FR · DE)  
**Origin:** Derived from the official [MIRACL](https://huggingface.co/datasets/miracl/miracl-corpus) benchmark corpus (Massive Information Retrieval Across Languages)

Each passage contains:

| Field | Description |
|---|---|
| `doc_id` | Unique corpus identifier |
| `title` | Wikipedia article title |
| `text` | Passage text |
| `language` | ISO 639-1 code (`en`, `es`, `fr`, `de`) |
| `url` | Source Wikipedia URL |
| `miracl_docid` | Original MIRACL passage ID (for qrel mapping) |

---

## What Was Built

### 1. Preprocessing Pipeline (`preprocessor.py`)

- Unicode NFKC normalisation → lower-case → strip URLs, HTML tags, standalone numbers
- NLTK word tokenisation
- Per-language stopword removal (NLTK lists + domain-specific additions for ES/FR/DE)
- Stemming: Porter (EN), Snowball (ES/FR/DE)
- Inverted index construction: `{term: [(doc_id, raw_count), …]}`

### 2. TF-IDF Vector Space Model (`search_engine.py`)

- Per-language inverted index with length-normalised TF × IDF weighting
- Cosine similarity ranking restricted to candidate docs from inverted index (efficient)
- **Sentence-aware snippet generation:** top 1–3 sentences ranked by query term overlap, joined with ` … `
- **Query term highlighting:** HTML `<mark>` tags over stemmed tokens and original surface words

### 3. LaBSE Cross-Lingual Retrieval (`cross_lingual.py`)

- Model: [`sentence-transformers/LaBSE`](https://huggingface.co/sentence-transformers/LaBSE) — 768-dim shared multilingual embedding space
- All 400K documents encoded as `title + text[:500]` → L2-normalised float32 vectors
- Stored in `doc_embeddings.npz` (~1.1 GB); loaded at startup
- Query in any language retrieves across all 4 languages via dot-product similarity — no translation required
- Per-language sub-results returned alongside the full multilingual ranking

### 4. Text Mining (`text_mining.py`)

| Technique | Implementation | Role |
|---|---|---|
| Document Clustering | KMeans (k=12) on TF-IDF + LSA (TruncatedSVD, 100 components) | Groups semantically related EN documents |
| Topic Modelling | LDA (sklearn, 10 topics) on CountVectorizer | Surfaces latent topics across the corpus |
| Query Expansion | Top terms from nearest cluster injected into query | Broadens TF-IDF recall |
| Keyphrase Extraction | Per-document TF-IDF top-N terms | Identifies representative keyphrases per document |

### 5. Evaluation (`evaluator.py`)

- 25 official MIRACL dev-set queries: 10 EN + 5 ES + 5 FR + 5 DE
- Relevance judgments from official MIRACL qrels via HuggingFace Hub
- Metrics: MAP, P@10, R@10, F1@10, nDCG@10
- Comparative evaluation in `MIRACL_IR_Evaluation.ipynb`

---

## Web Search Features

The Streamlit UI exposes three core web search features:

1. **Ranked top-10 retrieval** — results ordered by cosine similarity score (TF-IDF for VSM, dot-product for LaBSE)
2. **Sentence-aware snippet generation** — extracts the 1–3 most query-relevant sentences from each document body
3. **Highlighted query terms** — query words highlighted in yellow (`<mark>`) in both title and snippet for fast scanning

The UI displays both retrieval methods side by side:
- **Left panel:** TF-IDF VSM (monolingual retrieval) with optional query expansion tags
- **Right panel:** LaBSE cross-lingual retrieval with per-language sub-tabs (All · EN · ES · FR · DE)

---

## Architecture

```
Query
  │
  ├─► Preprocessor (normalise · tokenise · stopwords · stem)
  │         │
  │         ▼
  │   TF-IDF VSM ──► Inverted Index ──► Cosine Similarity ──► Top-10 (monolingual)
  │         │
  │         └─► Query Expansion (nearest cluster top-terms) ──► re-rank VSM
  │
  └─► LaBSE Encoder ──► Embedding Matrix (400K × 768) ──► Dot-Product ──► Top-10 (cross-lingual)
```

---

## Repository Structure

```
.
├── app.py                        # Streamlit UI — two-panel search interface
├── corpus_loader.py              # Corpus + eval query loading (cache-aware, /data/ aware)
├── preprocessor.py               # Tokenisation, stemming, inverted index builder
├── search_engine.py              # TF-IDF VSM: cosine retrieval, snippets, highlighting
├── cross_lingual.py              # LaBSE: encode, index, search (mono + cross-lingual)
├── text_mining.py                # DocumentClusterer, TopicModeller, KeyphraseExtractor
├── evaluator.py                  # IRMetrics: P@k, R@k, AP, MAP, F1@k, nDCG@k
├── build_embeddings.py           # One-time LaBSE pre-computation (memmap, ~1 h on MPS)
├── MIRACL_IR_Evaluation.ipynb    # Full evaluation notebook (39 cells)
├── IR_Assignment2_Formatted.pdf  # Assignment report
├── miracl_corpus_cache_queries.json  # 25 MIRACL dev-set evaluation queries
├── requirements.txt
├── Dockerfile
└── .streamlit/
    └── config.toml
```

---

## Evaluation Notebook

`MIRACL_IR_Evaluation.ipynb` is a self-contained evaluation notebook (39 cells) covering:

1. Corpus loading and language/length distribution plots
2. Preprocessing pipeline with tokenisation examples per language
3. Inverted index statistics and top-20 term frequency chart
4. TF-IDF VSM retrieval with example searches
5. Text mining: cluster sizes, top-terms table, LDA topic heatmap, query expansion demo, keyphrases
6. LaBSE embedding loading or on-the-fly computation + cross-lingual demo
7. 30 evaluation queries generated from corpus document titles (8 EN + 7 ES + 8 FR + 7 DE)
8. Per-query metrics (P@10, R@10, AP, nDCG@10) across 4 methods
9. 7 visualisation plots: MAP bar, grouped metrics, per-language breakdown, P@k curve, expansion impact, AP scatter, nDCG heatmap
10. Conclusions with method comparison

---

## How to Reproduce

### Prerequisites

```bash
pip install -r requirements.txt
```

Set a HuggingFace read token (required to download corpus and model weights):

```bash
export HF_TOKEN=hf_your_token_here
```

---

### Step 1 — Build the Corpus Cache

Downloads 100K passages per language (400K total) from `thoshiths/miracl-multilingual-4M` and official MIRACL dev qrels, then writes a local JSONL cache.

```bash
python corpus_loader.py
```

Outputs:
- `miracl_corpus_cache.jsonl` — 400K documents (~225 MB)
- `miracl_corpus_cache_queries.json` — 25 evaluation queries with mapped corpus IDs

> Skip if cache files already exist — `load_miracl()` reads from cache automatically.

---

### Step 2 — Pre-compute LaBSE Embeddings

Encodes all 400K documents into 768-dim LaBSE vectors using a numpy memmap (never holds the full matrix in RAM).

```bash
python build_embeddings.py
```

Output:
- `doc_embeddings.npz` — 400K × 768 float32 embeddings (~1.1 GB)

Expected time:
- Apple M-series (MPS): ~1 hour
- NVIDIA GPU (CUDA): ~20–30 min
- CPU only: ~4–6 hours

> Skip if `doc_embeddings.npz` already exists.

---

### Step 3 — Run the App Locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. First load preprocesses and indexes the corpus (~30–60 s for 400K docs); subsequent loads are instant via `@st.cache_resource`.

---

### Step 4 — Run the Evaluation Notebook

Open `MIRACL_IR_Evaluation.ipynb` in Jupyter. Requires the corpus cache and optionally `doc_embeddings.npz` (will compute if absent). Run all cells top to bottom.

```bash
jupyter notebook MIRACL_IR_Evaluation.ipynb
```

---

### Step 5 — Deploy to HuggingFace Spaces

The app is containerised with Docker:

1. Create a HuggingFace Space with **Docker SDK**
2. Push all source `.py` files, `Dockerfile`, `requirements.txt`, `README.md`, `.streamlit/config.toml`
3. Store large data files using HF Spaces persistent storage:
   ```bash
   hf sync ./data hf://buckets/<your-space-storage>
   ```
4. Add `HF_TOKEN` as a Space secret
5. The Space builds the Docker image and starts Streamlit on port 7860

The app resolves data paths: `/data/` (persistent storage) → local directory, so the bucket files are used automatically.

---

## Stack

| Component | Library / Model |
|---|---|
| UI | Streamlit 1.35 |
| Cross-lingual embeddings | `sentence-transformers/LaBSE` via HuggingFace Transformers |
| Sparse retrieval | Custom TF-IDF + inverted index |
| Clustering | scikit-learn KMeans + TruncatedSVD |
| Topic modelling | scikit-learn LDA |
| Stemming | NLTK (Snowball: ES/FR/DE, Porter: EN) |
| Corpus | HuggingFace Datasets (`miracl/miracl-corpus`) |
| Deployment | Docker on HuggingFace Spaces |
