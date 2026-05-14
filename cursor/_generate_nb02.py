"""Emit NLP_Workshop_02_TFIDF_and_Ranked_Retrieval.ipynb (standalone lab; same contract as NB01)."""
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
        r"""# NLP Workshop — Notebook 2  
## TF‑IDF and Ranked Retrieval

*Cursor assisted — this workshop track is drafted and maintained with Cursor for consistency and reproducibility.*

**Corpus:** English translations of the Quran (CSV)  
**Translation column (unchanged across the series):** `daryabadi`

---

## Where you are in the roadmap

```text
Keyword Search  →  Bag of Words  →  TF‑IDF  →  N‑grams  →  Embeddings  →  Semantic Search  →  LLM + Retrieval
                                      ▲
                               YOU ARE HERE
```

**Notebook 1** gave you counts: every word token in a verse contributed raw frequency. That is honest, but it often **over-rewards** words that appear everywhere in the corpus (for example highly repeated religious vocabulary).

**Notebook 2** adds the classic IR fix: **TF‑IDF weighting** so retrieval ranks documents using *importance*, not just *overlap*.

---

## Learning outcomes

By the end of this lab you should be able to:

1. Explain why raw BoW counts can mis-rank documents for search.  
2. Define **term frequency (TF)** and **inverse document frequency (IDF)** in words and with a simple formula.  
3. Build a **TF‑IDF vector space** with `sklearn` and run **cosine-ranked retrieval**.  
4. Compare **BoW vs TF‑IDF** rankings on the same queries and interpret differences.  
5. Read a **document-frequency / IDF table** for selected tokens and connect it to ranking behavior.  
6. Name what TF‑IDF still cannot do (meaning, order, negation nuance) and why **Notebook 3+** exists.  

**Prerequisite:** Notebook 1 (or equivalent comfort with preprocessing + cosine similarity).

---
"""
    )
)

cells.append(
    md(
        r"""# Part A — What goes wrong with BoW for retrieval?

BoW represents each verse as a vector of **non-negative integer counts**. Cosine similarity then asks: *how aligned are these two count vectors?*

That sounds reasonable until you notice two structural problems:

## Problem 1 — “Frequent everywhere” looks important

If a token appears in **many** verses, it creates **lots of opportunities** for accidental overlap between unrelated verses.

Example intuition: if nearly every verse mentions a small set of very common theological words, then two verses can look “similar” simply because they share **generic** vocabulary—not because they share the user’s *intent*.

## Problem 2 — raw counts reward length and repetition

Longer verses can accumulate larger counts for many tokens, which can skew similarity unless you choose a representation that down-weights uninformative repetition.

---

## The core idea of TF‑IDF

TF‑IDF keeps the bag-of-words skeleton, but replaces raw counts with **weighted** scores:

- **TF (term frequency):** how much a term appears *in this document* (variants exist: raw count, normalized count, log-scaled frequency).  
- **IDF (inverse document frequency):** how rare/informative a term is **across the corpus** (common terms get lower IDF; rare terms get higher IDF).

Then (conceptually):

```text
TFIDF(term, document) = TF(term, document) × IDF(term)
```

A term is “important” for a document if it is **locally frequent** *and* **globally distinctive**.

> **Note:** libraries differ in smoothing and log bases. `sklearn`’s implementation is consistent and reproducible; the pedagogical point is the *shape* of the behavior: down-weight ubiquitous terms, up-weight discriminative terms.

---
"""
    )
)

cells.append(
    md(
        r"""# Part B — Imports and plotting

| Library | Role |
|---|---|
| `pandas` / `numpy` | tables + numerics |
| `nltk` | same tokenizer/stopword contract as Notebook 1 |
| `sklearn` | `CountVectorizer`, `TfidfVectorizer`, cosine similarity |
| `matplotlib` / `seaborn` | compare models visually |

This notebook is **standalone**: it repeats the preprocessing contract from Notebook 1 so you can run it without rerunning NB01.
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
print("OK: imports loaded.")
"""
    )
)

cells.append(
    md(
        r"""# Part C — Configuration (same as Notebook 1)

```python
text_column = "daryabadi"
```
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

print("Rows:", len(df))
print("CSV:", CSV_PATH.resolve())
"""
    )
)

cells.append(
    md(
        r"""# Part D — Preprocessing contract (match Notebook 1)

We reuse the same cleaning steps so **Notebook 1 vs Notebook 2** comparisons are about **representation**, not accidental tokenizer drift.
"""
    )
)

cells.append(
    code(
        r"""STOP = set(stopwords.words("english"))
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
print("Example processed verse:\n", corpus[0][:220])
"""
    )
)

cells.append(
    md(
        r"""# Part E — TF and IDF (classical definitions + what sklearn does)

## Term frequency (TF)

Let \(f_{t,d}\) be the raw count of token \(t\) in document \(d\).

Common TF choices:

- **Raw:** \(\mathrm{TF}(t,d)=f_{t,d}\)  
- **Normalized:** divide by total tokens in \(d\) so long documents are not automatically “louder”  
- **Log:** \(\log(1+f_{t,d})\) to damp huge repeats

`TfidfVectorizer` supports these via parameters like `norm`, `sublinear_tf`.

## Inverse document frequency (IDF)

Let \(N\) be the number of documents and \(n_t\) the number of documents containing term \(t\).

A common textbook form is:

\[
\mathrm{IDF}(t)=\log \frac{N}{n_t}
\]

**Intuition:** if a word appears in *almost every verse*, \(\log(N/n_t)\) is near zero → it is not selective.  
If a word appears in only a handful of verses, IDF is larger → it is more selective.

## sklearn smoothing (read this once; then trust `idf_`)

`sklearn` uses a smoothed IDF definition so **zero document frequencies** do not explode. You can inspect the fitted values in `vectorizer.idf_`.

What you should remember for practice:

- **Higher IDF** ≈ more discriminative in this corpus (under your preprocessing).  
- **Lower IDF** ≈ very common across verses.

---

## The TF‑IDF score (conceptual)

\[
\mathrm{TFIDF}(t,d)=\mathrm{TF}(t,d)\times \mathrm{IDF}(t)
\]

After stacking all terms, each document becomes a **sparse vector** of TF‑IDF weights — still a bag of words in structure, but much better behaved for ranked retrieval than raw counts for many corpora.
"""
    )
)

cells.append(
    md(
        r"""## Pause — discussion prompts (answer in notes or in class)

1. If `"allah"` appears in far more verses than `"camel"`, which token should contribute **less** to “what makes this verse distinctive”? Why?  
2. Why can **cosine similarity on raw counts** still rank very generic verses highly for many user queries?  
3. Which failure mode from **Notebook 1** (for example `kindness` vs `mercy`) will TF‑IDF **still not solve**?

---

## `sklearn` knobs that change behavior (good to know early)

| Setting | What we use here | Why it matters |
|---|---|---|
| `smooth_idf` | default `True` | Stabilizes IDF when a term’s document frequency is tiny or oddities appear at corpus edges. |
| `sublinear_tf` | `True` | Applies `1 + log(tf)` so repeating a word gives **diminishing returns** (common in IR-style setups). |
| `norm='l2'` | default | Makes vectors unit length after weighting so cosine similarity behaves predictably. |

You can later rerun cells with `sublinear_tf=False` to feel how sensitive retrieval is to TF scaling.

---
"""
    )
)

cells.append(
    md(
        r"""# Part F — Fit BoW and TF‑IDF on the same vocabulary settings

We fit:

- `CountVectorizer` → matrix `X_count` (raw counts; good baseline)  
- `TfidfVectorizer` → matrix `X_tfidf` (weighted; search default)

We keep `lowercase=False` because we already lowercased in `preprocess_text`.
"""
    )
)

cells.append(
    code(
        r"""VEC_KW = dict(
    lowercase=False,
    token_pattern=r"(?u)\b\w\w+\b",
    min_df=1,
)

count_vec = CountVectorizer(**VEC_KW)
tfidf_vec = TfidfVectorizer(**VEC_KW, sublinear_tf=True)

X_count = count_vec.fit_transform(corpus)
X_tfidf = tfidf_vec.fit_transform(corpus)

terms_count = count_vec.get_feature_names_out()
terms_tfidf = tfidf_vec.get_feature_names_out()

assert list(terms_count) == list(terms_tfidf)

print("Documents:", X_count.shape[0])
print("Vocabulary size:", X_count.shape[1])
print("Count matrix nnz:", X_count.nnz)
print("TF-IDF matrix nnz:", X_tfidf.nnz)
"""
    )
)

cells.append(
    md(
        r"""## Interlude — document frequency vs IDF (connecting tables to intuition)

**Document frequency** \(df(t)\): how many verses contain token \(t\) at least once (under our vectorizer).

**IDF** in sklearn is stored in `tfidf_vec.idf_` and is **not** exactly \(\log(N/df)\) because of smoothing—but the **monotonic story** remains: ubiquitous terms have smaller IDF, rare terms have larger IDF.

This mirrors teaching patterns used in other workshop drafts: print a small table *before* staring only at retrieval outputs.
"""
    )
)

cells.append(
    code(
        r"""terms = tfidf_vec.get_feature_names_out()
idf = tfidf_vec.idf_
name_to_idx = {t: i for i, t in enumerate(terms)}

# Document frequency: number of verses with count >= 1
df_per_term = np.asarray((X_count > 0).sum(axis=0)).ravel()
N = X_count.shape[0]

interest = [
    "allah",
    "lord",
    "mercy",
    "merciful",
    "forgiveness",
    "guidance",
    "punishment",
    "day",
    "resurrection",
    "camel",
    "patience",
    "fire",
    "kindness",
]

rows = []
for tok in interest:
    j = name_to_idx.get(tok)
    if j is None:
        continue
    rows.append({"token": tok, "doc_freq": int(df_per_term[j]), "idf_sklearn": float(idf[j])})

stat_tbl = pd.DataFrame(rows).sort_values("idf_sklearn")
display(stat_tbl)

lo = np.argsort(idf)[:8]
hi = np.argsort(idf)[-8:]

print("\nLowest IDF (very common in this corpus under our preprocessing):")
for j in lo:
    print(f"  {terms[j]:18s}  df={int(df_per_term[j]):5d}  idf={idf[j]:.4f}")

print("\nHighest IDF (very rare / distinctive in this corpus):")
for j in hi[::-1]:
    print(f"  {terms[j]:18s}  df={int(df_per_term[j]):5d}  idf={idf[j]:.4f}")
"""
    )
)

cells.append(
    md(
        r"""## Visualization — IDF: common vs discriminative tokens

We plot IDF for a hand-picked mix of **likely-common** and **likely-rare** tokens (if present in the vocabulary). This connects the formula to what you see in retrieval.
"""
    )
)

cells.append(
    code(
        r"""idf = tfidf_vec.idf_
name_to_idx = {t: i for i, t in enumerate(terms_tfidf)}

demo_tokens = [
    "allah",
    "lord",
    "mercy",
    "believers",
    "charity",
    "orphan",
    "astray",
    "pharaoh",
    "adultery",
]

rows = []
for tok in demo_tokens:
    j = name_to_idx.get(tok)
    if j is None:
        continue
    rows.append((tok, float(idf[j])))

idf_demo = pd.DataFrame(rows, columns=["token", "idf"]).sort_values("idf")

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=idf_demo, x="idf", y="token", ax=ax, palette="mako")
ax.set_title("Smoothed IDF (sklearn) for selected tokens (if present)")
ax.set_xlabel("idf weight")
plt.tight_layout()
plt.show()

display(idf_demo)
"""
    )
)

cells.append(
    md(
        r"""## Mini worked example — TF × IDF “by hand” on three toy sentences

Other workshop drafts compute TF‑IDF **from scratch** before sklearn. This course does not make you implement a full sparse pipeline, but you *should* see one tiny numeric example so the product is not a black box.

Below we use a textbook-style **non-smoothed** IDF: \(\mathrm{IDF}(t)=\log_{10}(N/df_t)\) with \(N=3\) toy documents. (sklearn will differ numerically—this is for intuition only.)
"""
    )
)

cells.append(
    code(
        r"""import math

toy_docs = [
    "allah allah mercy",
    "allah day judgment",
    "camel merchant reward",
]
N = len(toy_docs)

# crude tokenization consistent with spirit of the lab (split on spaces)
tokens_per_doc = [d.split() for d in toy_docs]
vocab = sorted({t for toks in tokens_per_doc for t in toks})

# df: number of toy docs containing the token
from collections import Counter


def df_tok(tok: str) -> int:
    return sum(1 for toks in tokens_per_doc if tok in set(toks))


rows = []
for d_idx, toks in enumerate(tokens_per_doc):
    tf_raw = Counter(toks)
    for tok in vocab:
        tf = tf_raw.get(tok, 0)
        df = max(df_tok(tok), 1)
        idf10 = math.log10(N / df)
        rows.append({"doc": f"d{d_idx+1}", "token": tok, "tf": tf, "df": df, "idf_log10(N/df)": idf10, "tf_x_idf": tf * idf10})

toy_tbl = pd.DataFrame(rows)
# show only rows with tf>0 for readability
display(toy_tbl[toy_tbl["tf"] > 0].sort_values(["doc", "tf_x_idf"], ascending=[True, False]).reset_index(drop=True))

print("Toy vocabulary:", vocab)
"""
    )
)

cells.append(
    md(
        r"""## Visualization — distribution of sklearn IDF values across the full vocabulary

This histogram (inspired by fuller workshop drafts) shows how extreme the IDF range can be: a giant pile of “common-ish” terms and a long tail of rare terms.
"""
    )
)

cells.append(
    code(
        r"""fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(idf, bins=60, kde=False, ax=ax, color="teal")
ax.set_title("Distribution of sklearn smoothed IDF across all vocabulary terms")
ax.set_xlabel("idf")
ax.set_ylabel("count of terms")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md(
        r"""# Part G — TF‑IDF search engine (and a BoW baseline for comparison)

## TF‑IDF search engine (pipeline)

```text
User query
   ↓
Same preprocessing as Notebook 1
   ↓
TfidfVectorizer.transform (query vector)
   ↓
Cosine similarity vs all verse vectors
   ↓
Top-k ranked verses
```

We also keep a **BoW (CountVectorizer) engine** with the same tokenization settings so you can compare rankings **fairly**.

---

## Side-by-side engines

Both engines:

1. preprocess the user query with the **same** `preprocess_text`  
2. vectorize with the fitted vectorizer  
3. compute cosine similarity against all verses  
4. return **top‑k** results with scores

Cosine similarity is still the standard first choice for sparse lexical vectors because it mitigates raw length effects compared to a plain dot product.
"""
    )
)

cells.append(
    code(
        r"""def search_topk(vectorizer, X_matrix, query: str, k: int = 10) -> pd.DataFrame:
    q = preprocess_text(query, remove_stopwords=True, apply_stemming=USE_STEMMING)
    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, X_matrix).ravel()

    top_idx = np.argsort(-sims)[:k]
    out = df.loc[top_idx, ["verse_id", "text_raw", "text_processed"]].copy()
    out.insert(0, "rank", np.arange(1, k + 1))
    out["cosine_similarity"] = sims[top_idx]
    out["query_processed"] = q
    return out


def compare_engines(query: str, k: int = 8) -> None:
    bow = search_topk(count_vec, X_count, query, k=k)
    tfidf = search_topk(tfidf_vec, X_tfidf, query, k=k)

    print("QUERY:", repr(query))
    print("\n--- BoW (raw counts) ---")
    display(bow[["rank", "verse_id", "cosine_similarity", "text_raw"]])

    print("\n--- TF-IDF (weighted) ---")
    display(tfidf[["rank", "verse_id", "cosine_similarity", "text_raw"]])


# Example from the workshop spec (semantic-ish wording; still lexical retrieval)
compare_engines("allah is merciful", k=8)
"""
    )
)

cells.append(
    md(
        r"""## More comparisons — `mercy and forgiveness` and Notebook 1’s `kindness` probe

Other workshop versions often add **multi-word** queries like `mercy and forgiveness` to show re-ranking when multiple content words overlap.

They also revisit **`kindness`** as a humility check: TF‑IDF improves *weighting*, but it does **not** invent synonym axes. If `mercy` and `kindness` remain different vocabulary dimensions, many “meaning-level” failures from Notebook 1 can still happen.
"""
    )
)

cells.append(
    code(
        r"""compare_engines("mercy and forgiveness", k=6)

print("\n" + "=" * 72 + "\n")
compare_engines("kindness", k=6)
"""
    )
)

cells.append(
    md(
        r"""# Part H — Head-to-head: a query where generic overlap can dominate BoW

Try a query that contains **very common** words. BoW cosine can rise from shared “function-like content words” that are still frequent in this corpus even after stopword removal.

TF‑IDF often (not always) re-ranks so that passages sharing **more discriminative** terms move up.

> Student exercise: change the query string and watch how often ranks **cross** between the two tables.
"""
    )
)

cells.append(
    code(
        r"""compare_engines("lord of the worlds believers", k=8)
"""
    )
)

cells.append(
    md(
        r"""# Part I — Inspect TF‑IDF weights for one verse (interpretability)

Unlike raw counts, TF‑IDF weights can be read as “what terms this document emphasizes *relative to the corpus*”.

Below we print the top weighted tokens for a single verse row.
"""
    )
)

cells.append(
    code(
        r"""row_idx = 2
row = df.loc[row_idx]

v = X_tfidf[row_idx]
coo = v.tocoo()
pairs = [(j, val) for j, val in zip(coo.col, coo.data)]
pairs_sorted = sorted(pairs, key=lambda t: (-t[1], t[0]))

print("verse_id:", row["verse_id"])
print("RAW:\n", row["text_raw"][:350], "\n")

print("Top TF-IDF dimensions (token, weight):")
for j, val in pairs_sorted[:18]:
    print(f"  {terms_tfidf[j]:18s}  {val:.4f}")
"""
    )
)

cells.append(
    md(
        r"""## Visualization — same verse: BoW counts vs TF‑IDF weights (side-by-side)

Workshop drafts such as Claude’s NB02 often plot **raw counts** next to **TF‑IDF weights** for the *same* verse and the *same* token dimensions. The goal is visual: you should see rare-in-corpus words “pop” more under TF‑IDF than under raw counting.
"""
    )
)

cells.append(
    code(
        r"""ri = 2
vc = np.asarray(X_count[ri].todense()).ravel()
vt = np.asarray(X_tfidf[ri].todense()).ravel()

idx = np.where((vc > 0) | (vt > 0))[0]
idx = idx[np.argsort(-vt[idx])[:14]]
labels = [terms_tfidf[j] for j in idx]

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
axes[0].barh(labels[::-1], vc[idx][::-1], color="steelblue")
axes[0].set_title("BoW counts (subset of active dims)")
axes[0].set_xlabel("count")

axes[1].barh(labels[::-1], vt[idx][::-1], color="darkorange")
axes[1].set_title("TF-IDF weights (same dims)")
axes[1].set_xlabel("weight")

plt.suptitle(f"Verse {df.loc[ri, 'verse_id']}: counts vs TF-IDF", y=1.02)
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md(
        r"""## Visualization — rank drift for one query (BoW vs TF‑IDF)

We plot the **rank position** (1 is best) for the union of verse_ids appearing in either top‑8 list. This makes “rank swaps” visible at a glance.
"""
    )
)

cells.append(
    code(
        r"""QUERY = "charity orphans feeding poor"


def ranks_for(vectorizer, X_mat, query: str, k: int = 8):
    tbl = search_topk(vectorizer, X_mat, query, k=k)
    ids = tbl["verse_id"].tolist()
    return {vid: r for r, vid in enumerate(ids, start=1)}


rb = ranks_for(count_vec, X_count, QUERY, k=8)
rt = ranks_for(tfidf_vec, X_tfidf, QUERY, k=8)

ids = sorted(set(rb) | set(rt), key=lambda vid: min(rb.get(vid, 99), rt.get(vid, 99)))

plot_rows = []
for vid in ids:
    plot_rows.append({"verse_id": vid, "model": "BoW", "rank": rb.get(vid, np.nan)})
    plot_rows.append({"verse_id": vid, "model": "TF-IDF", "rank": rt.get(vid, np.nan)})

pr = pd.DataFrame(plot_rows)

fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(ids))))
sns.scatterplot(data=pr, x="rank", y="verse_id", hue="model", ax=ax, s=90)
ax.invert_xaxis()
ax.set_title(f"Rank comparison (lower rank is better): {QUERY!r}")
ax.set_xlabel("rank in top-8")
plt.tight_layout()
plt.show()

compare_engines(QUERY, k=8)
"""
    )
)

cells.append(
    md(
        r"""# Part J — What TF‑IDF does **not** fix (bridge to Notebook 3)

TF‑IDF is still fundamentally a **bag of words** model:

- **No semantics:** `mercy` and `compassion` are different dimensions unless they co-occur in patterns your query can latch onto.  
- **Limited context:** word order is mostly ignored (unigram TF‑IDF). Negation and subtle syntax remain fragile.  
- **Corpus-specific IDF:** the word “informative” depends on the collection; IDF is not universal truth.

That is why the next notebook adds **n‑grams** (local word order) as the next historically plausible upgrade.

```text
Notebook 2  →  TF-IDF weighting
Notebook 3  →  n-grams (local context)
```

---

## Mini checklist (self-assessment)

- [ ] I can explain why IDF down-weights ubiquitous tokens.  
- [ ] I can run ranked retrieval with `TfidfVectorizer` + cosine similarity.  
- [ ] I can describe at least one query where BoW and TF‑IDF rankings disagree—and why that happened.  

When you are ready, open **Notebook 3 — N‑grams and Context**.
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

out_path = Path(__file__).parent / "NLP_Workshop_02_TFIDF_and_Ranked_Retrieval.ipynb"
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote:", out_path)

if __name__ == "__main__":
    print("Done.")
