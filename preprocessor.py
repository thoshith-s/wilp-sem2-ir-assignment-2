"""
preprocessor.py
Multilingual text preprocessing pipeline for the MIRACL corpus (EN, ES, FR, DE).
Handles tokenisation, stopword removal, and Snowball/Porter stemming per language.
Builds the inverted index with TF scores used by the TF-IDF search engine.

Author  : Thoshith S
"""

import re
import unicodedata
import nltk

nltk.download("punkt",       quiet=True)
nltk.download("punkt_tab",   quiet=True)
nltk.download("stopwords",   quiet=True)

from nltk.corpus import stopwords as nltk_sw
from nltk.stem import PorterStemmer, SnowballStemmer

_en_stemmer = PorterStemmer()
_es_stemmer = SnowballStemmer("spanish")
_fr_stemmer = SnowballStemmer("french")
_de_stemmer = SnowballStemmer("german")

# ── Stopword lists ─────────────────────────────────────────────────────────────
ENGLISH_STOPWORDS = set(nltk_sw.words("english")) | {
    "also", "one", "two", "three", "known", "used", "given",
    "however", "although", "including", "several", "many", "other",
    "first", "second", "new", "old", "made", "may", "like",
}

SPANISH_STOPWORDS = set(nltk_sw.words("spanish")) | {
    "también", "así", "sino", "aunque", "durante", "través",
}

FRENCH_STOPWORDS = set(nltk_sw.words("french")) | {
    "aussi", "ainsi", "dont", "lors", "jusqu", "comme",
}

GERMAN_STOPWORDS = set(nltk_sw.words("german")) | {
    "auch", "sowie", "jedoch", "dabei", "durch", "beim",
}

_STOPWORDS = {
    "en": ENGLISH_STOPWORDS,
    "es": SPANISH_STOPWORDS,
    "fr": FRENCH_STOPWORDS,
    "de": GERMAN_STOPWORDS,
}

_STEMMERS = {
    "en": _en_stemmer,
    "es": _es_stemmer,
    "fr": _fr_stemmer,
    "de": _de_stemmer,
}


# ── Regex helpers ──────────────────────────────────────────────────────────────
_URL_RE   = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE  = re.compile(r"<[^>]+>")
_NUM_RE   = re.compile(r"\b\d+\b")
# Allow letters from Latin + Latin Extended (covers ES/FR/DE accents + ß)
_LATIN_RE = re.compile(r"[^a-záéíóúüñàâæçèêëîïôùûüÿœæœÄÖÜß\s]", re.IGNORECASE)


# ── Core tokenizer (Latin-script languages: EN, ES, FR, DE) ───────────────────

def tokenize_latin(text: str, lang: str = "en") -> list[str]:
    """
    Normalize → clean → NLTK tokenize → filter stopwords → Snowball/Porter stem.
    Works for EN, ES, FR, DE.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NUM_RE.sub(" ", text)
    text = _LATIN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = nltk.word_tokenize(text)
    stops   = _STOPWORDS.get(lang, ENGLISH_STOPWORDS)
    stemmer = _STEMMERS.get(lang, _en_stemmer)

    result = []
    for t in tokens:
        if len(t) < 2 or t in stops:
            continue
        stemmed = stemmer.stem(t)
        if len(stemmed) >= 2:
            result.append(stemmed)
    return result


def tokenize_english(text: str) -> list[str]:
    return tokenize_latin(text, "en")


def tokenize_for_lang(text: str, lang: str) -> list[str]:
    """Dispatch to the right tokenizer by language code."""
    return tokenize_latin(text, lang if lang in _STEMMERS else "en")


# ── Document preprocessing ─────────────────────────────────────────────────────

def preprocess_document(doc: dict) -> dict:
    lang = doc.get("language", "en")
    text = doc.get("text", "")
    tokens = tokenize_for_lang(text, lang)
    return {
        "doc_id":   doc["doc_id"],
        "title":    doc["title"],
        "tokens":   tokens,
        "raw_text": text,
        "language": lang,
        "url":      doc.get("url", ""),
    }


def preprocess_corpus(corpus: list[dict]) -> list[dict]:
    print("Preprocessing corpus …")
    result = []
    for i, doc in enumerate(corpus):
        result.append(preprocess_document(doc))
        if (i + 1) % 50 == 0 or (i + 1) == len(corpus):
            print(f"  {i+1}/{len(corpus)} documents processed", end="\r")
    print(f"  {len(result)}/{len(corpus)} documents processed")
    return result


# ── Inverted index ─────────────────────────────────────────────────────────────

def build_inverted_index(preprocessed_corpus: list[dict]) -> tuple[dict, dict, set]:
    """
    Build inverted index from preprocessed corpus.

    Returns:
        inverted_index : {term: [(doc_id, raw_count), …]}
        doc_lengths    : {doc_id: number_of_tokens}
        vocab          : set of all unique terms
    """
    print("Building inverted index …")
    inverted_index: dict[str, list[tuple[str, int]]] = {}
    doc_lengths:    dict[str, int]  = {}
    vocab:          set[str]        = set()

    for pdoc in preprocessed_corpus:
        doc_id = pdoc["doc_id"]
        tokens = pdoc["tokens"]
        doc_lengths[doc_id] = max(len(tokens), 1)

        tf_counts: dict[str, int] = {}
        for t in tokens:
            tf_counts[t] = tf_counts.get(t, 0) + 1
            vocab.add(t)

        for term, count in tf_counts.items():
            inverted_index.setdefault(term, []).append((doc_id, count))

    print(f"  Vocabulary size : {len(vocab):,}")
    print(f"  Index entries   : {sum(len(v) for v in inverted_index.values()):,}")
    return inverted_index, doc_lengths, vocab


if __name__ == "__main__":
    for lang, text in [
        ("en", "The quick brown fox jumps over the lazy dog."),
        ("es", "El zorro marrón rápido salta sobre el perro perezoso."),
        ("fr", "Le renard brun rapide saute par-dessus le chien paresseux."),
        ("de", "Der schnelle braune Fuchs springt über den faulen Hund."),
    ]:
        tokens = tokenize_for_lang(text, lang)
        print(f"[{lang}] {tokens}")
