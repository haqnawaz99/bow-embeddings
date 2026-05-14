"""Emit NLP_Workshop_03_Ngrams_and_Context.ipynb — depth aligned with codex/NLP_Workshop_03."""
import json
import uuid
from pathlib import Path


def md(s: str):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True), "id": str(uuid.uuid4())}


def code(s: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": s.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
        "id": str(uuid.uuid4()),
    }


cells: list = []

cells.append(
    md(
        r"""# NLP Workshop — Notebook 3  
## N‑grams and Local Context

*Cursor assisted — this workshop track is drafted and maintained with Cursor for consistency and reproducibility.*

**Corpus:** English translations of the Quran (CSV)  
**Translation column:** `daryabadi` (unchanged across the series)

---

## Where you are in the roadmap

```text
Keyword Search  →  Bag of Words  →  TF‑IDF  →  N‑grams  →  Embeddings  →  Semantic Search  →  LLM + Retrieval
                                                    ▲
                                             YOU ARE HERE
```

Notebook 2 improved **weighting** (TF‑IDF), but **unigram** models still mostly ignore **word order**. This notebook adds **n‑grams**—and goes through **trigrams**—so you can see the classic tradeoff: richer local context vs **vocabulary explosion** and **sparsity**.

---

## Learning outcomes (Codex-aligned depth)

By the end of this lab you should be able to:

1. Explain **unigrams, bigrams, and trigrams** and how they relate to a sliding window over tokens.  
2. Show where **order matters** (`not good` vs `very good`; optional collapse under stopwording).  
3. Fit **three** TF‑IDF spaces: `(1,1)`, `(1,2)`, and `(1,3)` on the same preprocessed corpus.  
4. Compare **retrieval** across all three models on the same queries and interpret rank / score changes.  
5. Discuss **limitations** (semantics, sparsity, short context) and why **Notebook 4 (Word2Vec)** comes next.

**Prerequisites:** Notebooks 1–2.

---
"""
    )
)

cells.append(
    md(
        r"""# Part A — Why Notebook 2 was still not enough

TF‑IDF fixes “common words dominate raw counts,” but **unigram TF‑IDF** still treats a verse roughly as a multiset of isolated tokens.

That breaks down for phrases such as:

```text
day of judgment
children of israel
straight path
```

**N‑grams** keep short consecutive sequences as features so phrase fragments can survive as dimensions (subject to preprocessing).

### Historical motivation

N‑grams are a classical way to add **short-range structure** without leaving the vector-space paradigm. They do **not** deliver deep semantics—that is why embeddings follow.

---
"""
    )
)

cells.append(
    md(
        r"""# Part B — Imports, data, preprocessing (same contract as NB01–02)

| Library | Role |
|---|---|
| `sklearn` | `CountVectorizer`, `TfidfVectorizer`, cosine similarity |
| `pandas` / `numpy` | tables + numerics |
| `nltk` | tokenization, `ngrams`, stopwords |
| `matplotlib` / `seaborn` | vocabulary growth + score plots |

---
"""
    )
)

cells.append(
    code(
        r"""from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    display = print

for pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find("corpora/stopwords" if pkg == "stopwords" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

try:
    get_ipython().run_line_magic("matplotlib", "inline")  # type: ignore[name-defined]
except Exception:
    pass

sns.set_theme(style="whitegrid", context="notebook")

print("Python:", sys.version.split()[0])
"""
    )
)

cells.append(
    code(
        r"""text_column = "daryabadi"

_CANDIDATES = [
    Path("quran_translations.csv"),
    Path("../quran_translations.csv"),
    Path.cwd() / "quran_translations.csv",
    Path.cwd().parent / "quran_translations.csv",
]

CSV_PATH = next((p for p in _CANDIDATES if p.exists()), None)
if CSV_PATH is None:
    raise FileNotFoundError("Could not find quran_translations.csv.")

df = pd.read_csv(CSV_PATH, keep_default_na=False)
if text_column not in df.columns:
    raise KeyError(text_column)

df["verse_id"] = df["Surah"].astype(str) + ":" + df["Verse"].astype(str)
df["text_raw"] = df[text_column].astype(str)

STOP = set(stopwords.words("english"))
PUNCT_RE = re.compile(r"[^a-z0-9\s']+", re.IGNORECASE)
USE_STEMMING = False


def preprocess_text(
    text: str,
    *,
    remove_stopwords: bool = True,
    apply_stemming: bool = False,
) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    s = str(text).lower()
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()

    tokens = word_tokenize(s)
    tokens = [t.strip("'") for t in tokens if t.strip("'")]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP and t != ""]

    if apply_stemming:
        from nltk.stem import PorterStemmer

        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


df["text_processed"] = df["text_raw"].apply(
    lambda x: preprocess_text(x, remove_stopwords=True, apply_stemming=USE_STEMMING)
)
corpus = df["text_processed"].tolist()

print("Verses:", len(corpus))
print("CSV:", CSV_PATH.resolve())
print("Sample processed:\n", corpus[0][:200])
"""
    )
)

cells.append(
    md(
        r"""# Part C — Sliding window intuition (one verse)

Before `sklearn`, it helps to **list** n‑grams explicitly for a few tokens.

Below: take one preprocessed verse as a token list, then print unigrams, bigrams, and trigrams (as tuples).
"""
    )
)

cells.append(
    code(
        r"""sample = corpus[100]
toks = sample.split()

print("verse_id:", df.loc[100, "verse_id"])
print("tokens (first 18):", toks[:18])
print()

print("Sample unigrams (first 12):", list(ngrams(toks, 1))[:12])
print("Sample bigrams (first 10):", list(ngrams(toks, 2))[:10])
print("Sample trigrams (first 8):", list(ngrams(toks, 3))[:8])
"""
    )
)

cells.append(
    md(
        r"""# Part D — Toy: `not good` vs `very good` (CountVectorizer)

Same toy as before: two short English sentences. We compare `(1,1)` vs `(1,2)` **without** sklearn English `stop_words` so `not` survives for teaching.

Then we show what happens when we run the **workshop** `preprocess_text` (NLTK stopwords remove `not`).
"""
    )
)

cells.append(
    code(
        r"""toy_sentences = [
    "This movie is not good",
    "This movie is very good",
]

vec_uni = CountVectorizer(lowercase=True, ngram_range=(1, 1), token_pattern=r"(?u)\b\w\w+\b")
vec_bi = CountVectorizer(lowercase=True, ngram_range=(1, 2), token_pattern=r"(?u)\b\w\w+\b")

Xu = vec_uni.fit_transform(toy_sentences)
Xb = vec_bi.fit_transform(toy_sentences)

cos_uni = cosine_similarity(Xu[0], Xu[1])[0, 0]
cos_bi = cosine_similarity(Xb[0], Xb[1])[0, 0]
print(f"Cosine (unigrams only): {cos_uni:.4f}")
print(f"Cosine (uni+bigrams):   {cos_bi:.4f}")

for s in toy_sentences:
    print(repr(s), "->", repr(preprocess_text(s)))
"""
    )
)

cells.append(
    md(
        r"""## Stopwords vs phrases (Codex-style reminder)

If you drop words like `of`, phrases change shape:

```text
day of judgment  →  day judgment   (after aggressive stopword removal)
```

Sometimes that is still useful. Sometimes it **weakens** the very n‑gram signal you wanted.

---
"""
    )
)

cells.append(
    code(
        r"""demo_q = "day of judgment"
s = str(demo_q).lower()
s = PUNCT_RE.sub(" ", s)
s = re.sub(r"\s+", " ", s).strip()
with_sw = word_tokenize(s)
no_sw = preprocess_text(demo_q).split()

print("Query:", demo_q)
print("Tokens (stopwords kept in this demo cell):", with_sw)
print("Tokens after workshop preprocess (joined):", preprocess_text(demo_q))
"""
    )
)

cells.append(
    md(
        r"""# Part E — Build **three** TF‑IDF models (unigram, +bigram, +trigram)

We compare:

1. `(1, 1)` — unigrams only (closest to Notebook 2 structure)  
2. `(1, 2)` — unigrams + bigrams  
3. `(1, 3)` — unigrams + bigrams + trigrams  

All share the same `token_pattern`, `min_df=1`, and `sublinear_tf=True` so the experiment isolates **n‑gram range**.

> Note: `(1, 3)` can be large in memory/time. If a machine struggles, reduce the corpus for practice or raise `min_df` slightly for a *separate* sensitivity experiment.
"""
    )
)

cells.append(
    code(
        r"""VEC_KW = dict(
    lowercase=False,
    token_pattern=r"(?u)\b\w\w+\b",
    min_df=1,
    sublinear_tf=True,
)

tfidf_uni = TfidfVectorizer(ngram_range=(1, 1), **VEC_KW)
tfidf_bi = TfidfVectorizer(ngram_range=(1, 2), **VEC_KW)
tfidf_tri = TfidfVectorizer(ngram_range=(1, 3), **VEC_KW)

X_uni = tfidf_uni.fit_transform(corpus)
X_bi = tfidf_bi.fit_transform(corpus)
X_tri = tfidf_tri.fit_transform(corpus)

terms_uni = tfidf_uni.get_feature_names_out()
terms_bi = tfidf_bi.get_feature_names_out()
terms_tri = tfidf_tri.get_feature_names_out()

print("Unigram matrix:           ", X_uni.shape)
print("Unigram+bigram matrix:    ", X_bi.shape)
print("Up-to-trigram matrix:     ", X_tri.shape)

vocab_sizes = pd.DataFrame(
    {
        "representation": ["(1,1) unigram", "(1,2) uni+bi", "(1,3) up to tri"],
        "vocabulary_size": [len(terms_uni), len(terms_bi), len(terms_tri)],
        "nonzero_entries": [X_uni.nnz, X_bi.nnz, X_tri.nnz],
    }
)
display(vocab_sizes)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=vocab_sizes, x="representation", y="vocabulary_size", ax=ax, palette="Blues_r")
ax.set_title("Vocabulary growth as n-gram range widens")
ax.set_ylabel("number of features")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md(
        r"""# Part F — Retrieval helpers (three engines)

Same recipe as Notebook 2: preprocess query → `transform` → cosine similarity → top‑k.

We attach a `model` label so results concatenate cleanly for plots.
"""
    )
)

cells.append(
    code(
        r"""def search_topk(vectorizer, X_matrix, query: str, label: str, k: int = 5) -> pd.DataFrame:
    q = preprocess_text(query, remove_stopwords=True, apply_stemming=USE_STEMMING)
    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, X_matrix).ravel()
    top_idx = np.argsort(-sims)[:k]

    out = df.loc[top_idx, ["verse_id", "text_raw", "text_processed"]].copy()
    out.insert(0, "rank", np.arange(1, k + 1))
    out["score"] = sims[top_idx]
    out["model"] = label
    out["query_processed"] = q
    return out


def search_uni(q: str, k: int = 5):
    return search_topk(tfidf_uni, X_uni, q, "Unigram TF-IDF", k=k)


def search_bi(q: str, k: int = 5):
    return search_topk(tfidf_bi, X_bi, q, "Uni+Bigram TF-IDF", k=k)


def search_tri(q: str, k: int = 5):
    return search_topk(tfidf_tri, X_tri, q, "Up-to-Trigram TF-IDF", k=k)


def show_three(query: str, k: int = 5) -> None:
    print("QUERY:", repr(query))
    print("\n--- Unigram ---")
    display(search_uni(query, k=k)[["rank", "verse_id", "score", "text_raw"]])
    print("\n--- Uni + Bigram ---")
    display(search_bi(query, k=k)[["rank", "verse_id", "score", "text_raw"]])
    print("\n--- Up to Trigram ---")
    display(search_tri(query, k=k)[["rank", "verse_id", "score", "text_raw"]])


show_three("day of judgment", k=5)
"""
    )
)

cells.append(
    md(
        r"""# Part G — Inspect which n‑gram features a query activates

Interpretability: print the **non-zero** query dimensions for `(1,1)` vs `(1,2)` (dense slice of the sparse query vector is fine here because only one query row).
"""
    )
)

cells.append(
    code(
        r"""inspect_query = "day of judgment"
qc = preprocess_text(inspect_query)

qv1 = tfidf_uni.transform([qc]).toarray().ravel()
qv2 = tfidf_bi.transform([qc]).toarray().ravel()

u_table = pd.DataFrame({"feature": terms_uni, "weight": qv1})
u_table = u_table[u_table["weight"] > 0].sort_values("weight", ascending=False)

b_table = pd.DataFrame({"feature": terms_bi, "weight": qv2})
b_table = b_table[b_table["weight"] > 0].sort_values("weight", ascending=False)

print("Preprocessed query:", repr(qc))
print("\nActive unigram features:")
display(u_table.head(20))

print("\nActive uni+bigram features (note bigrams contain a space):")
display(b_table.head(25))
"""
    )
)

cells.append(
    md(
        r"""# Part H — Multiple phrase queries (top‑3 under each model)

Same pattern as Codex: run several **phrase-like** queries and compare three retrieval tables each time.
"""
    )
)

cells.append(
    code(
        r"""comparison_queries = [
    "day of judgment",
    "straight path",
    "children of israel",
    "most merciful",
]

for q in comparison_queries:
    print("=" * 88)
    show_three(q, k=3)
"""
    )
)

cells.append(
    md(
        r"""# Part I — Rank drift across **three** representations

We take the union of verse_ids in the top‑8 lists for each model and plot rank (1 is best). This extends the Notebook 2 “rank drift” idea to **trigrams**.
"""
    )
)

cells.append(
    code(
        r"""RANK_QUERY = "day of judgment"
K = 8


def rank_map(search_fn, q, k):
    t = search_fn(q, k=k)
    return {r["verse_id"]: int(r["rank"]) for _, r in t.iterrows()}


r1 = rank_map(search_uni, RANK_QUERY, K)
r2 = rank_map(search_bi, RANK_QUERY, K)
r3 = rank_map(search_tri, RANK_QUERY, K)

ids = sorted(set(r1) | set(r2) | set(r3), key=lambda v: min(r1.get(v, 99), r2.get(v, 99), r3.get(v, 99)))

plot_rows = []
for vid in ids:
    plot_rows.append({"verse_id": vid, "model": "(1,1)", "rank": r1.get(vid, np.nan)})
    plot_rows.append({"verse_id": vid, "model": "(1,2)", "rank": r2.get(vid, np.nan)})
    plot_rows.append({"verse_id": vid, "model": "(1,3)", "rank": r3.get(vid, np.nan)})

pr = pd.DataFrame(plot_rows)

fig, ax = plt.subplots(figsize=(11, max(3.5, 0.32 * len(ids))))
sns.scatterplot(data=pr, x="rank", y="verse_id", hue="model", ax=ax, s=85)
ax.invert_xaxis()
ax.set_title(f"Rank drift (top-{K} union): {RANK_QUERY!r}")
ax.set_xlabel("rank (1 = best in that model's top-k)")
plt.tight_layout()
plt.show()

show_three(RANK_QUERY, k=5)
"""
    )
)

cells.append(
    md(
        r"""# Part J — Interpreting differences (when n‑grams help)

When retrieval improves, it is often because stable fragments appear as features, for example:

- `day judgment` (after `of` is removed) can still behave like a **phrase fragment** in bigram space  
- `straight path`, `children israel`, `mercy forgiveness` (after `and` removal) can score as multi-token units

This is still **lexical** retrieval—not semantic understanding.

---
"""
    )
)

cells.append(
    md(
        r"""# Part K — Discussion prompts (reflection / class notes)

1. Why might **bigrams** help more than **trigrams** on some queries?  
2. Why does adding trigrams usually **increase sparsity** and vocabulary size?  
3. Can n‑grams solve synonymy such as `kindness` vs `mercy`?  
4. Why might aggressive **stopword removal** weaken some phrase patterns you wanted n‑grams to catch?

---
"""
    )
)

cells.append(
    md(
        r"""# Part L — Limitations of n‑grams

### 1. No deep semantics

`mercy`, `compassion`, `benevolence` are still mostly different dimensions unless the corpus ties them in shared phrases.

### 2. Sparse feature explosion

You measured vocabulary growth—this is the main classical reason people moved toward **dense embeddings**.

### 3. Short context only

Even trigrams only see **local** windows; long-range dependencies remain hard.

### 4. Surface form dependence

Typos, morphology, and translation choices still matter.

---
"""
    )
)

cells.append(
    md(
        r"""# Part M — Failure query (semantic intent, lexical gaps)

Codex-style “humility check”: a query can sound **conceptual** while still being **lexically** mismatched to many verses.

Run the same failure query through all three models and compare.
"""
    )
)

cells.append(
    code(
        r"""failure_query = "kindness to parents"
show_three(failure_query, k=5)
"""
    )
)

cells.append(
    md(
        r"""# Part N — Visual comparison of top scores (one query, three models)

Bar chart: top‑5 hits per model with cosine score on the x‑axis (Codex-style). If verse ids repeat across models, hue distinguishes representation.
"""
    )
)

cells.append(
    code(
        r"""viz_query = "straight path"

u = search_uni(viz_query, k=5).copy()
b = search_bi(viz_query, k=5).copy()
t = search_tri(viz_query, k=5).copy()

u["representation"] = "(1,1)"
b["representation"] = "(1,2)"
t["representation"] = "(1,3)"

viz_df = pd.concat([u, b, t], ignore_index=True)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=viz_df, x="score", y="verse_id", hue="representation", ax=ax)
ax.set_title(f"Top retrieval scores by model — query: {viz_query!r}")
ax.set_xlabel("cosine similarity")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md(
        r"""# Part O — Summary, limitation map, transition to Notebook 4

### What we learned

- Unigrams ignore most order; bigrams/trigrams recover **short** local context.  
- TF‑IDF + n‑grams improves many **phrase-like** queries—but not all.  
- Cost: **bigger vocabulary**, sparser matrices, heavier computation.

### Historical pattern

```text
counting words was not enough,
weighting words was not enough,
so researchers preserved local word sequences,
and still needed dense meaning vectors.
```

### Limitation → next step

| n‑grams improve | Still weak | Next notebook |
|---|---|---|
| Local word order | Deep semantics | **Notebook 4 — Word2Vec** |
| Phrase-sensitive retrieval | Synonymy | **Embeddings** |
| More context than unigrams | Long-range reasoning | **Later: transformers / LLMs** |

---

## Mini self-check

- [ ] I can explain unigrams vs bigrams vs trigrams.  
- [ ] I can compare `(1,1)` vs `(1,2)` vs `(1,3)` retrieval on the same query.  
- [ ] I can explain why n‑grams do not solve synonymy.  
- [ ] I can explain why this motivates **Word2Vec**.

---

### Suggested exercises

1. Try queries such as `most forgiving`, `path of those`, `day of resurrection`.  
2. For each query, compare unigram vs bigram vs trigram top‑3.  
3. Find one query where bigrams clearly help; find one where **all three** still miss the intent.

When you are ready, open **Notebook 4 — Word Embeddings with Word2Vec**.
"""
    )
)

cells.append(
    code(
        r"""print("Notebook 3 complete (Codex-depth track: through trigrams).")
"""
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

out_path = Path(__file__).parent / "NLP_Workshop_03_Ngrams_and_Context.ipynb"
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote:", out_path)

if __name__ == "__main__":
    print("Done.")
