"""
app.py — MIRACL Multilingual Search Engine (Streamlit UI)

Author  : Thoshith S
Corpus  : MIRACL Wikipedia passages — EN · ES · FR · DE (400 K docs, 100 K / language)
Dataset : https://huggingface.co/datasets/thoshiths/miracl-multilingual-4M

Run:
    streamlit run app.py
"""

import os
import re
import html as _html

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LOKY_MAX_CPU_COUNT"]     = "1"

import streamlit as st

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MIRACL Multilingual Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
LANGUAGES = ["en", "es", "fr", "de"]
LANG_NAME = {"en": "English", "es": "Spanish", "fr": "French", "de": "German"}
DATASET_URL = "https://huggingface.co/datasets/thoshiths/miracl-multilingual-4M"

DEMO_QUERIES = [
    ("What is a civil war?",                       "en"),
    ("History of the Roman Empire",                "en"),
    ("¿Quién creó el lenguaje Java?",              "es"),
    ("¿Cuándo vivieron los celtas?",               "es"),
    ("Qui est Achille dans la guerre de Troie?",   "fr"),
    ("Qu'est-ce que la philosophie?",              "fr"),
    ("Wie ist die japanische Sprache entstanden?", "de"),
    ("Was ist Quantenmechanik?",                   "de"),
]

# ── Global stylesheet ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Result card ── */
.result-card {
    background: #1e1e2e;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    border-left: 3px solid #4a6fa5;
}
.result-card.clir {
    border-left-color: #c9963b;
}

/* ── Rank number ── */
.rank-num {
    display: inline-block;
    background: #4a6fa5;
    color: #fff;
    border-radius: 4px;
    width: 22px; height: 22px;
    text-align: center; line-height: 22px;
    font-size: 11px; font-weight: 700;
    margin-right: 6px;
}
.rank-num.clir-rank { background: #c9963b; }

/* ── Language badge ── */
.lang-badge {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 10px; font-weight: 700;
    letter-spacing: .04em;
    margin-left: 5px;
    font-family: monospace;
}
.en-badge { background: #1c3a2a; color: #5cb88a; }
.es-badge { background: #3a1c1c; color: #d96060; }
.fr-badge { background: #1c2a3a; color: #6aaed6; }
.de-badge { background: #3a2e1c; color: #d4a44c; }

/* ── Cross-lingual badge ── */
.clir-badge {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 10px; font-weight: 700;
    margin-left: 5px;
    background: #3a2e1c;
    color: #c9963b;
    border: 1px solid #c9963b44;
    font-family: monospace;
}

/* ── Text within cards ── */
.result-title { font-size: 14px; font-weight: 600; color: #c9d1f5; }
.score-text   { font-size: 11px; color: #6e6e8a; margin-left: 6px; }
.snippet-text {
    font-size: 12px; color: #b8bcd4;
    margin: 5px 0 0 0; line-height: 1.6;
}
.url-text { font-size: 11px; color: #555a7a; margin-top: 3px; }
.url-text a { color: #555a7a; text-decoration: none; }
.url-text a:hover { color: #7f9fd4; }

/* ── Highlighted query terms ── */
mark.qterm {
    background: #f0e14a;
    color: #111;
    border-radius: 2px;
    padding: 0 2px;
    font-weight: 600;
}

/* ── Section header for retrieval panels ── */
.panel-header {
    border-bottom: 2px solid #2e2e3e;
    padding-bottom: 8px;
    margin-bottom: 12px;
}
.panel-title {
    font-size: 13px; font-weight: 700;
    color: #c9d1f5; margin: 0;
    text-transform: uppercase; letter-spacing: .06em;
}
.panel-subtitle { font-size: 11px; color: #6e6e8a; margin: 3px 0 0 0; }

/* ── Web search features strip ── */
.wsf-strip {
    background: #16182a;
    border: 1px solid #2a2e45;
    border-radius: 6px;
    padding: 7px 14px;
    font-size: 11px; color: #7a80a0;
    margin: 6px 0 14px 0;
    display: flex; gap: 18px; flex-wrap: wrap; align-items: center;
}
.wsf-strip b { color: #9aa0c0; }

/* ── Query expansion row ── */
.exp-strip {
    font-size: 11px; color: #7a80a0;
    margin-bottom: 10px;
}
.exp-term {
    display: inline-block;
    background: #1e2640;
    color: #9aa0c0;
    border: 1px solid #2e3a5a;
    border-radius: 3px;
    padding: 1px 7px; font-size: 11px;
    margin: 2px 3px 2px 0;
    font-family: monospace;
}

/* ── Metadata text ── */
.meta-text { font-size: 11px; color: #555a7a; }
</style>
""", unsafe_allow_html=True)


# ── Helper: language badge ─────────────────────────────────────────────────────
def _lang_badge(lang: str) -> str:
    """Return a styled ISO-639-1 language code badge."""
    return f'<span class="lang-badge {lang}-badge">{lang.upper()}</span>'


def _clir_badge(src: str, tgt: str) -> str:
    """Return a cross-lingual indicator badge (src → tgt language codes)."""
    return f'<span class="clir-badge">{src.upper()} → {tgt.upper()}</span>'


# ── Helper: render ranked result cards ────────────────────────────────────────
def _render_results(
    results: list[dict],
    query_lang: str | None = None,
    is_clir: bool = False,
) -> None:
    """
    Render a ranked list of search results as HTML cards.

    Each card displays:
      - Rank number
      - Title (with highlighted query terms)
      - Language code badge and optional cross-lingual indicator
      - Similarity score
      - Sentence-aware snippet (with highlighted query terms)
      - Source URL

    Parameters
    ----------
    results    : List of result dicts produced by SearchEngine or
                 MultilingualEmbeddingRetrieval.
    query_lang : ISO-639-1 code of the query language; used to flag
                 cross-lingual hits.
    is_clir    : When True, applies the LaBSE colour scheme to rank badges.
    """
    if not results:
        st.caption("No results found.")
        return

    for r in results:
        lang     = r.get("language", "en")
        is_cross = (query_lang is not None) and (lang != query_lang)
        card_cls = "result-card clir" if is_cross else "result-card"
        rank_cls = "rank-num clir-rank" if is_clir else "rank-num"
        cross_b  = _clir_badge(query_lang, lang) if is_cross else ""

        snippet_html = r.get("highlighted_snippet") or _html.escape(r.get("snippet", ""))
        title_html   = r.get("highlighted_title")   or _html.escape(r.get("title", ""))

        url = r.get("url", "")
        url_html = (
            f'<div class="url-text">'
            f'<a href="{url}" target="_blank">{url[:80]}</a>'
            f'</div>'
        ) if url else ""

        st.markdown(f"""
<div class="{card_cls}">
  <span class="{rank_cls}">{r["rank"]}</span>
  <span class="result-title">{title_html}</span>
  {_lang_badge(lang)}{cross_b}
  <span class="score-text">score {r["score"]:.4f}</span>
  <p class="snippet-text">{snippet_html}</p>
  {url_html}
</div>""", unsafe_allow_html=True)


# ── Engine loader (cached across Streamlit sessions) ──────────────────────────
@st.cache_resource(show_spinner="Initialising search engine…")
def load_engine() -> tuple:
    """
    Load and initialise all IR components once per server process.

    Builds the TF-IDF inverted index, fits text-mining models (KMeans,
    LDA, keyphrase extractor) and loads pre-computed LaBSE embeddings.

    Returns
    -------
    tuple : (engine, clusterer, topic_modeller, kp_extractor, clir)
    """
    from corpus_loader import load_miracl
    from preprocessor  import preprocess_corpus, build_inverted_index
    from search_engine import SearchEngine
    from text_mining   import DocumentClusterer, TopicModeller, KeyphraseExtractor
    from cross_lingual import MultilingualEmbeddingRetrieval

    corpus, _    = load_miracl()
    prep         = preprocess_corpus(corpus)
    idx, dlen, _ = build_inverted_index(prep)
    engine       = SearchEngine(corpus, prep, idx, dlen)

    clusterer = DocumentClusterer(corpus, n_clusters=12)
    clusterer.fit()

    topic_modeller = TopicModeller(corpus, n_topics=10)
    topic_modeller.fit()

    kp_extractor = KeyphraseExtractor(corpus)
    kp_extractor.fit()

    clir = MultilingualEmbeddingRetrieval(corpus)
    _emb = next((p for p in ["/data/doc_embeddings.npz", "doc_embeddings.npz"]
                 if os.path.exists(p)), None)
    if _emb:
        clir.load_embeddings(_emb)
    else:
        st.warning(
            "doc_embeddings.npz not found. "
            "Run `python build_embeddings.py` to enable LaBSE cross-lingual search."
        )

    return engine, clusterer, topic_modeller, kp_extractor, clir


# ── Session state ──────────────────────────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""


def _set_query(q: str) -> None:
    """Store a query in session state and trigger a page rerun."""
    st.session_state.query = q


# ── Initialise ────────────────────────────────────────────────────────────────
try:
    engine, clusterer, topic_modeller, kp_extractor, clir = load_engine()
    stats  = engine.get_statistics()
    loaded = True
except Exception as _exc:
    loaded     = False
    load_error = str(_exc)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### MIRACL Search")
    st.caption("Multilingual Wikipedia Search Engine")
    st.divider()

    # Dataset reference
    st.markdown(
        f"**Dataset**  \n"
        f"[thoshiths/miracl-multilingual-4M]({DATASET_URL})  \n"
        f"40 K passages · EN / ES / FR / DE · Wikipedia"
    )
    st.divider()

    # Corpus statistics
    if loaded:
        st.markdown("**Corpus Statistics**")
        by_lang = stats.get("by_lang", {})
        c1, c2  = st.columns(2)
        c1.metric("Total docs", f"{stats['n_docs']:,}")
        c2.metric("Vocab size", f"{stats['vocab_size']:,}")
        for i, lang in enumerate(LANGUAGES):
            (c1 if i % 2 == 0 else c2).metric(
                lang.upper(), f"{by_lang.get(lang, 0):,}"
            )
        non_en_pct = 100 * (stats["n_docs"] - by_lang.get("en", 0)) / max(stats["n_docs"], 1)
        st.caption(f"Non-English: {non_en_pct:.1f}%")
        st.divider()

    # Search settings
    st.markdown("**Search Settings**")
    top_k = st.slider("Top-K results", min_value=5, max_value=20, value=10)

    expansion_mode = st.selectbox(
        "Query Expansion",
        ["None",
         "Clustering (KMeans + LSA)",
         "Topic Modelling (LDA)",
         "Keyphrase Extraction (TF-IDF)"],
        index=0,
        help="Appends extra terms to the TF-IDF query to improve recall.",
    )

    show_clir = st.checkbox("Show cross-lingual results (LaBSE)", value=True)
    st.divider()

    # Sample queries
    st.markdown("**Sample Queries**")
    for q_text, q_lang in DEMO_QUERIES:
        label = f"[{q_lang.upper()}]  {q_text}"
        if st.button(label, key=f"demo_{q_text}", use_container_width=True):
            _set_query(q_text)
            st.rerun()

    st.divider()

    # Retrieval methods description
    st.markdown("**Retrieval Methods**")
    st.markdown(
        "**TF-IDF VSM** — Normalized term frequency × inverse document "
        "frequency with cosine similarity. Monolingual; matched to the "
        "detected query language.  \n\n"
        "**LaBSE** — Language-agnostic BERT Sentence Embeddings (768-dim). "
        "Projects the query into a shared multilingual space and retrieves "
        "passages from all four languages."
    )
    st.divider()

    st.markdown("**Text Mining (Query Expansion)**")
    st.markdown(
        "- KMeans + LSA Clustering  \n"
        "- LDA Topic Modelling  \n"
        "- TF-IDF Keyphrase Extraction"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.title("MIRACL Multilingual Search")
st.caption(
    "Wikipedia passage retrieval across EN / ES / FR / DE  ·  "
    f"Corpus: [{DATASET_URL.split('/')[-1]}]({DATASET_URL})"
)
st.divider()

if not loaded:
    st.error(f"Engine failed to load: {load_error}")
    st.stop()


# ── Query input ────────────────────────────────────────────────────────────────
col_in, col_btn = st.columns([8, 1])
with col_in:
    query = st.text_input(
        "Search query",
        value=st.session_state.query,
        placeholder="Enter a query in English, Spanish, French, or German",
        label_visibility="collapsed",
    )
with col_btn:
    st.button("Search", use_container_width=True, type="primary")


# ── Welcome state ──────────────────────────────────────────────────────────────
if not query:
    st.markdown("""
**How it works**

Enter a query in any of the four supported languages.
Two retrieval panels are shown side by side:

| Panel | Method | Scope |
|---|---|---|
| Left | TF-IDF Vector Space Model | Monolingual — same language as query |
| Right | LaBSE Multilingual Embeddings | Cross-lingual — all four languages |

**Web search features:** ranked top-10 results · sentence-aware snippet
extraction · query term highlighting in titles and snippets.

Select a sample query from the sidebar to get started.
""")
    st.stop()


# ── Language detection ─────────────────────────────────────────────────────────
from cross_lingual import detect_language
detected = detect_language(query)
other    = [lg for lg in LANGUAGES if lg != detected]


# ── Web Search Features strip ─────────────────────────────────────────────────
query_surface = list(dict.fromkeys(re.findall(r'\b\w{3,}\b', query)))
terms_html    = " · ".join(
    f'<mark class="qterm">{_html.escape(w)}</mark>' for w in query_surface
)
st.markdown(
    f'<div class="wsf-strip">'
    f'<b>Web Search Features</b>'
    f'<span>Ranked top-{top_k} retrieval</span>'
    f'<span>Sentence-aware snippet generation</span>'
    f'<span>Highlighted query terms: {terms_html}</span>'
    f'<span>Detected language: <b>{LANG_NAME[detected]}</b> ({detected})</span>'
    f'</div>',
    unsafe_allow_html=True,
)


# ── Query expansion (computed here; displayed inside the VSM panel) ───────────
expanded_terms: list[str] = []
exp_label = ""

if expansion_mode == "Clustering (KMeans + LSA)":
    expanded_terms = clusterer.expand_query(query, top_n=5)
    exp_label      = "KMeans + LSA"
elif expansion_mode == "Topic Modelling (LDA)":
    expanded_terms = topic_modeller.expand_query(query, top_n=5)
    exp_label      = "LDA Topic"
elif expansion_mode == "Keyphrase Extraction (TF-IDF)":
    expanded_terms = kp_extractor.expand_query(query, top_n=5)
    exp_label      = "Keyphrase"


# ── LaBSE retrieval (run once before layout split) ────────────────────────────
cl_out: dict | None = None
if show_clir:
    if clir._is_fitted:
        with st.spinner("Running LaBSE cross-lingual retrieval…"):
            cl_out = clir.search_cross_lingual(
                query, source_lang=detected, top_k=top_k
            )
    else:
        st.info(
            "LaBSE embeddings unavailable. "
            "Run `python build_embeddings.py` to enable cross-lingual search."
        )


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS — TWO-PANEL LAYOUT
#   Left  : TF-IDF Vector Space Model  (monolingual)
#   Right : LaBSE Multilingual Embeddings  (cross-lingual)
# ══════════════════════════════════════════════════════════════════════════════
col_vsm, col_clir = st.columns(2, gap="large")


# ── Left panel: TF-IDF VSM ────────────────────────────────────────────────────
with col_vsm:
    # Build the subtitle line — include expansion method when active
    vsm_subtitle = (
        f"Normalized TF × IDF · cosine similarity · "
        f"monolingual ({LANG_NAME[detected]})"
    )
    if expanded_terms:
        exp_terms_inline = " ".join(
            f'<span class="exp-term">{_html.escape(t)}</span>'
            for t in expanded_terms
        )
        expansion_line = (
            f'<p class="panel-subtitle" style="margin-top:4px">'
            f'Query expansion ({exp_label}): {exp_terms_inline}'
            f'</p>'
        )
    else:
        expansion_line = ""

    st.markdown(
        f'<div class="panel-header">'
        f'<p class="panel-title">TF-IDF Vector Space Model</p>'
        f'<p class="panel-subtitle">{vsm_subtitle}</p>'
        f'{expansion_line}'
        f'</div>',
        unsafe_allow_html=True,
    )

    vsm_results = engine.search(
        query,
        language=detected,
        top_k=top_k,
        target_lang=detected,
        expanded_terms=expanded_terms or None,
    )

    if vsm_results:
        st.markdown(
            f'<p class="meta-text">{len(vsm_results)} results · '
            f'{LANG_NAME[detected]} ({detected}) corpus</p>',
            unsafe_allow_html=True,
        )
        _render_results(vsm_results, query_lang=detected, is_clir=False)
    else:
        st.info(
            "No keyword matches. "
            "Try enabling query expansion or broaden the search terms."
        )


# ── Right panel: LaBSE ────────────────────────────────────────────────────────
with col_clir:
    st.markdown(
        '<div class="panel-header">'
        '<p class="panel-title">LaBSE Multilingual Embeddings</p>'
        '<p class="panel-subtitle">'
        '768-dim shared multilingual space · Language-Agnostic BERT Sentence Embeddings'
        '</p></div>',
        unsafe_allow_html=True,
    )

    if cl_out is None:
        st.info("Cross-lingual retrieval is disabled. Enable it in the sidebar.")
    else:
        all_multi = cl_out.get("multilingual_results", [])
        foreign   = [r for r in all_multi if r["language"] != detected]
        n_cross   = len(foreign)

        st.markdown(
            f'<p class="meta-text">{len(all_multi)} results across all languages · '
            f'{n_cross} cross-lingual match(es)</p>',
            unsafe_allow_html=True,
        )

        # Language sub-tabs: All results + per-language breakdown
        lang_tabs = st.tabs(["All"] + [lg.upper() for lg in LANGUAGES])

        with lang_tabs[0]:
            _render_results(all_multi[:top_k], query_lang=detected, is_clir=True)

        for i, lg in enumerate(LANGUAGES):
            with lang_tabs[i + 1]:
                lang_res = cl_out.get(f"{lg}_results", [])
                if lang_res:
                    avg_sim = sum(r["score"] for r in lang_res) / len(lang_res)
                    note    = "cross-lingual" if lg != detected else "same language as query"
                    st.caption(f"avg. similarity {avg_sim:.4f} · {note}")
                    _render_results(
                        lang_res,
                        query_lang=detected,
                        is_clir=(lg != detected),
                    )
                else:
                    st.caption("No results for this language.")
