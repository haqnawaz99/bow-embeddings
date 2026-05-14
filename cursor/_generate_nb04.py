"""Emit NLP_Workshop_04_Word2Vec_Embeddings.ipynb — Word2Vec, CBOW vs Skip-gram, PCA/t-SNE."""
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
        r"""# NLP Workshop — Notebook 4  
## Word Embeddings with Word2Vec

*Cursor assisted — this workshop track is drafted and maintained with Cursor for consistency and reproducibility.*

**Corpus:** English translations of the Quran (CSV)  
**Translation column:** `daryabadi`

---

## Where you are in the roadmap

```text
… →  N‑grams  →  Word2Vec  →  FastText  →  Sentence embeddings  →  …
                      ▲
               YOU ARE HERE
```

**Notebooks 1–3** used **sparse** vectors: huge vocabularies, mostly zeros, and no shared structure between `mercy` and `compassion` unless they co-occur in the same tiny n‑gram patterns.

**Word2Vec** learns **dense** vectors (e.g. 100–300 dimensions) so that words with **similar contexts** become neighbors in vector space. That is a different idea from “count the same string.”

---

## Learning outcomes

1. Contrast **sparse bag-of-words / TF‑IDF** vs **dense word embeddings**.  
2. Describe **CBOW** vs **Skip‑gram** at a high level (what is predicted from what).  
3. Train **Word2Vec** on the Daryabadi verses and query **similar words**.  
4. Build **mean‑pooled verse vectors** and run **semantic verse retrieval** with cosine similarity (Claude-style demo).  
5. Compare **TF‑IDF vs Word2Vec** retrieval on the **`kindness`** query with a side‑by‑side score plot.  
6. Visualize a word subset with **PCA** and **t‑SNE** (2D).  
7. Explain remaining limits (OOV words, static vectors, no full sentence meaning) → **Notebook 5 (FastText)**.

**Prerequisites:** Notebooks 1–3.

---
"""
    )
)

cells.append(
    md(
        r"""# Part A — Why embeddings?

## Sparse classical vectors

In TF‑IDF (even with n‑grams), each dimension is usually a **lexical feature** (a word or n‑gram type). Two different words are two different dimensions unless your preprocessing merges them.

**Consequence:** “similar meaning, different spelling” is hard to represent.

## Dense embeddings

Word2Vec (2013) popularized the idea:

> A word is represented by a **short vector of real numbers**, learned so that **context predicts the word** (Skip‑gram) or **words predict the context** (CBOW).

Similar **distributional** behavior → similar vectors.

This is still **not** “true understanding,” but it is a historically crucial step toward modern semantic search and transformers.

---
"""
    )
)

cells.append(
    md(
        r"""# Part B — CBOW vs Skip‑gram (Mikolov et al.)

| Model | Sketch | Typical story |
|---|---|---|
| **CBOW** (`sg=0`) | Predict the **center** word from surrounding context | Slightly faster; smoother for frequent words |
| **Skip‑gram** (`sg=1`) | Predict **context** words from the center word | Often stronger for rare words / finer semantics |

Both learn **lookup tables** (rows = word vectors) by gradient-style optimization (implemented efficiently in `gensim` with negative sampling).

**Hyperparameters you will touch:**

- `vector_size`: embedding dimensionality (e.g. 100)  
- `window`: how far context reaches along the sentence  
- `min_count`: ignore very rare tokens (reduces noise; can hide words)  
- `epochs`: more passes over the data (often better, diminishing returns)

---
"""
    )
)

cells.append(
    md(
        r"""# Part C — Imports, data, token sentences for `gensim`

`gensim.models.Word2Vec` expects `sentences` as a **list of token lists**, one list per “sentence.” Here each **verse** is one sentence.

We reuse the **same** `preprocess_text` contract as earlier notebooks (lowercase, punctuation, NLTK tokenization, English stopword removal).
"""
    )
)

cells.append(
    code(
        r"""from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    display = print

warnings.filterwarnings("ignore", category=UserWarning)

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
print("OK: imports loaded.")
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
df["text_raw"] = df[text_column].astype(str)

STOP = set(stopwords.words("english"))
PUNCT_RE = re.compile(r"[^a-z0-9\s']+", re.IGNORECASE)


def preprocess_text(text: str) -> str:
    if not text:
        return ""
    s = str(text).lower()
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = word_tokenize(s)
    tokens = [t.strip("'") for t in tokens if t.strip("'")]
    tokens = [t for t in tokens if t not in STOP and t != ""]
    return " ".join(tokens)


# Aligned lists: same rows as Word2Vec training sentences (>=3 tokens)
verse_texts = []
verse_ids = []
sentences = []
for _, row in df.iterrows():
    raw = str(row["text_raw"])
    s = preprocess_text(raw)
    toks = s.split()
    if len(toks) >= 3:
        verse_texts.append(raw)
        verse_ids.append(f"{row['Surah']}:{row['Verse']}")
        sentences.append(toks)

print("Verses with >=3 tokens after preprocess:", len(sentences))
print("Example verse_id:", verse_ids[0])
print("Example sentence (first 15 tokens):", sentences[0][:15])
"""
    )
)

cells.append(
    md(
        r"""# Part D — Train two Word2Vec models (CBOW vs Skip‑gram)

We train **two** small models on the same sentences so you can compare neighbors. Training time is kept moderate with `vector_size=100`, `epochs=8`, and `min_count=3`.

> If `mercy` is below `min_count` on your machine after you change settings, lower `min_count` or increase epochs.
"""
    )
)

cells.append(
    code(
        r"""W2V_KW = dict(
    vector_size=100,
    window=5,
    min_count=3,
    workers=4,
    epochs=8,
    seed=42,
)

model_cbow = Word2Vec(sentences=sentences, sg=0, **W2V_KW)
model_skip = Word2Vec(sentences=sentences, sg=1, **W2V_KW)

print("CBOW   vocab size:", len(model_cbow.wv))
print("SkipGM vocab size:", len(model_skip.wv))
"""
    )
)

cells.append(
    md(
        r"""# Part E — Semantic neighbors with `most_similar`

Workshop demo word: **`mercy`** (if present in vocabulary).

If a `KeyError` appears, pick another word from `model_skip.wv.index_to_key[:50]`.
"""
    )
)

cells.append(
    code(
        r"""def show_neighbors(label: str, model: Word2Vec, word: str, topn: int = 12):
    if word not in model.wv:
        print(f"{word!r} not in vocabulary (try lowering min_count or another word).")
        return
    print(label, "neighbors for", repr(word))
    for w, sc in model.wv.most_similar(word, topn=topn):
        print(f"  {w:18s}  {sc:.4f}")


for label, m in [("CBOW", model_cbow), ("Skip-gram", model_skip)]:
    print("=" * 48, label, "=" * 48)
    show_neighbors(label, m, "mercy")
    print()
"""
    )
)

cells.append(
    md(
        r"""## Optional — similarity scores between word pairs

Cosine similarity in `gensim` is accessed via `wv.similarity(w1, w2)` when both words exist.
"""
    )
)

cells.append(
    code(
        r"""pairs = [("mercy", "forgiveness"), ("mercy", "punishment"), ("lord", "mercy")]

for w1, w2 in pairs:
    if w1 in model_skip.wv and w2 in model_skip.wv:
        print(f"skip-gram sim({w1!r}, {w2!r}) = {model_skip.wv.similarity(w1, w2):.4f}")
    else:
        print("skip:", w1, w2, "(missing)")
"""
    )
)

cells.append(
    md(
        r"""# Part F — Pick a vocabulary subset for 2D plots

t‑SNE is **nonlinear** and can distort global distances; use it as an **exploration** plot. **PCA** is linear and easier to interpret as “first two principal components.”

We take the **top frequent** words in the Skip‑gram vocabulary plus a few hand-picked words (if present), up to `MAX_WORDS`.
"""
    )
)

cells.append(
    code(
        r"""MAX_WORDS = 120
HAND = ["mercy", "forgiveness", "guidance", "prayer", "fire", "paradise", "allah", "lord", "day", "night"]

freq = model_skip.wv.index_to_key[:MAX_WORDS]
words = []
seen = set()
for w in list(freq) + HAND:
    if w in model_skip.wv and w not in seen:
        words.append(w)
        seen.add(w)

print("Number of plotted words:", len(words))
mat = np.stack([model_skip.wv[w] for w in words], axis=0)
print("Matrix shape:", mat.shape)
"""
    )
)

cells.append(
    md(
        r"""# Part G — PCA projection (2D)
"""
    )
)

cells.append(
    code(
        r"""pca = PCA(n_components=2, random_state=42)
xy_pca = pca.fit_transform(mat)

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(xy_pca[:, 0], xy_pca[:, 1], alpha=0.55, s=35)
for i, w in enumerate(words):
    if i % 2 == 0 or w in set(HAND):
        ax.annotate(w, (xy_pca[i, 0], xy_pca[i, 1]), fontsize=7, alpha=0.85)
ax.set_title("Word2Vec (Skip-gram) — PCA to 2D (annotated subset)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.tight_layout()
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
"""
    )
)

cells.append(
    md(
        r"""# Part H — t‑SNE projection (2D)

t‑SNE is slower; we run on the **same** `mat` with a fixed `random_state` for reproducibility. Adjust `perplexity` if you change `MAX_WORDS` a lot.
"""
    )
)

cells.append(
    code(
        r"""tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=min(30, max(5, len(words) // 4)),
    max_iter=1000,
    init="pca",
)
xy_tsne = tsne.fit_transform(mat)

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(xy_tsne[:, 0], xy_tsne[:, 1], alpha=0.55, s=35, c="darkgreen")
for i, w in enumerate(words):
    if i % 2 == 0 or w in set(HAND):
        ax.annotate(w, (xy_tsne[i, 0], xy_tsne[i, 1]), fontsize=7, alpha=0.85)
ax.set_title("Word2Vec (Skip-gram) — t-SNE to 2D (annotated subset)")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md(
        r"""# Part I — Mean‑pooled verse vectors + semantic search (Claude-style retrieval)

Word2Vec gives **word** vectors. To search **verses**, a classic hack is **mean pooling**: average the vectors of all in-vocabulary words in the verse.

- Use the **same** representation for the **query string** (after preprocessing into tokens).  
- Rank verses by **cosine similarity** between the query vector and each verse vector.

This mirrors the **“semantic search engine”** idea from Claude’s NB04: show that **`kindness`** can retrieve mercy‑related verses even when the match is not “exact keyword logic.”
"""
    )
)

cells.append(
    code(
        r"""def verse_to_vector_mean(tokens: list[str], model: Word2Vec) -> np.ndarray:
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if not vecs:
        return np.zeros(model.vector_size)
    return np.mean(np.stack(vecs, axis=0), axis=0)


print("Building mean-pooled verse embeddings (Skip-gram model)...")
verse_vectors_mean = np.array([verse_to_vector_mean(t, model_skip) for t in sentences])
print("Shape (verses, dims):", verse_vectors_mean.shape)

# TF-IDF on the same tokenized verses (joined), comparable to NB02 style
corpus_joined = [" ".join(t) for t in sentences]
tfidf_retrieval = TfidfVectorizer(
    lowercase=False,
    token_pattern=r"(?u)\b\w\w+\b",
    min_df=1,
    sublinear_tf=True,
)
X_tfidf_verses = tfidf_retrieval.fit_transform(corpus_joined)
print("TF-IDF matrix for verses:", X_tfidf_verses.shape)


def semantic_search_w2v(query: str, top_k: int = 5):
    q_toks = preprocess_text(query).split()
    q_vec = verse_to_vector_mean(q_toks, model_skip)
    if float(np.linalg.norm(q_vec)) == 0.0:
        print("Query vector is zero — no query words in Word2Vec vocabulary.")
        return []
    sims = cosine_similarity(q_vec.reshape(1, -1), verse_vectors_mean)[0]
    top_idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i]), verse_texts[i], verse_ids[i]) for i in top_idx]


def tfidf_search(query: str, top_k: int = 5):
    q_str = preprocess_text(query)
    qv = tfidf_retrieval.transform([q_str])
    sims = cosine_similarity(qv, X_tfidf_verses)[0]
    top_idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i]), verse_texts[i], verse_ids[i]) for i in top_idx]


def display_search_results(title: str, results: list) -> None:
    print(title)
    print("=" * 72)
    if not results:
        print("(no results)")
        return
    for rank, (i, score, text, vid) in enumerate(results, 1):
        snippet = (text[:120] + "...") if len(text) > 120 else text
        print(f"Rank {rank}  score={score:.4f}  {vid}")
        print(" ", snippet)
        print()


print("QUERY 1: 'kindness' (recall NB01 exact-match limits)")
print("Word2Vec retrieval uses mean-pooled vectors + cosine similarity.\n")
res_k = semantic_search_w2v("kindness", top_k=5)
display_search_results("Word2Vec semantic search — query: 'kindness'", res_k)

print("\nQUERY 2: 'divine compassion' (multi-word query)")
res_c = semantic_search_w2v("divine compassion", top_k=5)
display_search_results("Word2Vec semantic search — query: 'divine compassion'", res_c)

print("\nQUERY 3: 'punishment of the wicked'")
res_p = semantic_search_w2v("punishment of the wicked", top_k=5)
display_search_results("Word2Vec semantic search — query: 'punishment of the wicked'", res_p)
"""
    )
)

cells.append(
    md(
        r"""# Part J — TF‑IDF vs Word2Vec on **`kindness`** (same corpus rows)

We run **lexical** TF‑IDF retrieval (sparse bag on the joined preprocessed verse strings) vs **dense** mean‑pooled Word2Vec retrieval, both with cosine similarity.

Read the scores and snippets critically: TF‑IDF can still rank verses that share weighted tokens with the query; Word2Vec can rank verses whose **average word geometry** is close to the query—even when overlap is imperfect. Neither is “truth,” both are **models**.
"""
    )
)

cells.append(
    code(
        r"""cmp_query = "kindness"

tfidf_results = tfidf_search(cmp_query, top_k=5)
w2v_results = semantic_search_w2v(cmp_query, top_k=5)

print("Query:", repr(cmp_query))
print("\n--- TF-IDF (lexical vector space) ---")
for rank, (i, score, text, vid) in enumerate(tfidf_results, 1):
    snippet = (text[:100] + "...") if len(text) > 100 else text
    print(f"  Rank {rank}  [{score:.4f}]  {vid}  {snippet}")

print("\n--- Word2Vec (mean-pooled dense vectors) ---")
for rank, (i, score, text, vid) in enumerate(w2v_results, 1):
    snippet = (text[:100] + "...") if len(text) > 100 else text
    print(f"  Rank {rank}  [{score:.4f}]  {vid}  {snippet}")

# Side-by-side score bars (Claude-style figure)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
labels = [f"R{r}" for r in range(1, len(tfidf_results) + 1)]
ax1.bar(labels, [s for _, s, _, _ in tfidf_results], color="#FF7043", edgecolor="black")
ax1.set_title("TF-IDF top-5 scores\n(query preprocessed)")
ax1.set_ylabel("cosine similarity")
ax1.set_ylim(0, 1)

ax2.bar(labels[: len(w2v_results)], [s for _, s, _, _ in w2v_results], color="#42A5F5", edgecolor="black")
ax2.set_title("Word2Vec top-5 scores\n(mean-pooled query vs verses)")
ax2.set_ylabel("cosine similarity")
ax2.set_ylim(0, 1)

plt.suptitle('TF-IDF vs Word2Vec: query = "kindness"', fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print("\nKey insight (interpret, do not memorize):")
print("  If TF-IDF scores are strong, the query tokens still align with lexical features.")
print("  Word2Vec can re-rank using geometry of related words learned from context.")
"""
    )
)

cells.append(
    md(
        r"""# Part K — What Word2Vec still does **not** fix (bridge to FastText)

| Limitation | Why it matters | Next notebook |
|---|---|---|
| **OOV / rare morphs** | No vector for unseen or mistyped words | **FastText** (subword info) |
| **Static vectors** | One vector per word type; “bank” river vs money not separated unless contexts differ a lot | Contextual embeddings later |
| **Sentence meaning** | Word2Vec is word-level; mean pooling is only a rough whole-verse vector | **Sentence-BERT (NB06)** |
| **Domain shift** | Vectors reflect *this* corpus’s co-occurrence statistics | Fine-tune or use broader corpora |

```text
Notebook 4  →  Word2Vec (dense word vectors)
Notebook 5  →  FastText (subword + rare words)
```

---

## Mini self-check

- [ ] I can explain CBOW vs Skip‑gram in one paragraph.  
- [ ] I can interpret `most_similar` output as “shared local contexts.”  
- [ ] I can contrast TF‑IDF vs mean‑pooled Word2Vec retrieval on one example query.  
- [ ] I can explain one weakness that motivates FastText.

When you are ready, open **Notebook 5 — FastText and Subword Embeddings**.
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

out_path = Path(__file__).parent / "NLP_Workshop_04_Word2Vec_Embeddings.ipynb"
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote:", out_path)

if __name__ == "__main__":
    print("Done.")
