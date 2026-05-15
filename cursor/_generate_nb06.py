"""Emit NLP_Workshop_06_Sentence_Embeddings_and_Semantic_Search.ipynb — SBERT-style encoders vs classical retrieval."""
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
        r"""# NLP Workshop — Notebook 6  
## Sentence Embeddings and Semantic Search

*Cursor assisted — this workshop track is drafted and maintained with Cursor for consistency and reproducibility.*

**Corpus:** English translations of the Quran (CSV)  
**Translation column:** `daryabadi`

---

## Where you are in the roadmap

```text
NB01  Preprocess + BoW  →  NB02  TF‑IDF  →  NB03  N‑grams  →  NB04  Word2Vec  →  NB05  FastText  →  NB06  Sentence embeddings  →  NB07+  FAISS / RAG
                                                                                                                    ▲
                                                                                                             YOU ARE HERE
```

**Notebooks 4–5** built **word-level** vectors (and mean-pooling hacks for whole verses). A verse is not “a bag of independent words,” but pooling treats it that way.

**Sentence embeddings** (popularized in teaching by **Sentence-BERT**–style *bi-encoders*) map an **entire string** to one vector in a single forward pass through a **Transformer** backbone. The model can condition each token on its **context**, then pool to a fixed size (often **CLS** or mean over tokens).

---

## Learning outcomes

1. Contrast **mean-pooled word vectors** vs **sentence encoder** vectors for the same text.  
2. Encode verses with a **pretrained** `sentence-transformers` model and run **cosine similarity retrieval**.  
3. Compare **dense semantic retrieval** to **TF‑IDF** on the **same verses** and **same queries**.  
4. Discuss **latency, model size, domain shift**, and when pretrained sentence models help or mislead.  
5. Name what **Notebook 7** will add (vector search at scale, e.g. **FAISS**).  
6. Read a **sentence–sentence similarity** table on hand-written **STS-style** pairs (cosine in embedding space).  
7. Interpret a **PCA** projection of a sample of verse embeddings.  
8. Read a **top‑10 score bar chart** for one dense query.

**Prerequisites:** Notebooks 1–5.

---
"""
    )
)

cells.append(
    md(
        r"""# Part A — Why move from “words” to “sentences”?

## What mean pooling missed

Mean pooling over Word2Vec / FastText ignores **order**, **negation**, and **multi-word expressions** unless those signals happen to align with average directions in space.

## What a sentence encoder adds

A Transformer-based encoder builds **contextual** hidden states: the vector for `bank` can differ in *river bank* vs *money bank* (depending on architecture and training). After pooling, you get **one vector per input string**, tuned on sentence-level training objectives (e.g. natural language inference, paraphrase pairs).

## Honest limitations (preview)

- **Domain shift:** a model trained mostly on web / Wikipedia English may not match religious register perfectly.  
- **Cost:** heavier than TF‑IDF; first download of weights needs disk + network.  
- **Not truth:** high similarity is still **model similarity**, not theological or factual endorsement.

---
"""
    )
)

cells.append(
    md(
        r"""# Part B — `sentence-transformers` in one picture

We use a small English model (**`all-MiniLM-L6-v2`**, 384 dimensions) so laptops can run the lab.

```text
verse text string  --->  [Transformer encoder]  --->  normalize  --->  vector in R^384
query string       --->  [same encoder]           --->  normalize  --->  vector in R^384
```

Retrieval: **cosine similarity** between the query vector and every verse vector (same idea as NB02, but in a **dense** space learned from large supervised / self-supervised pretraining).

---
"""
    )
)

cells.append(
    md(
        r"""# Part C — Imports, data, and verse list (same filter as NB04–NB05)

We keep **`preprocess_text`** for the **TF‑IDF** baseline. For **sentence embeddings**, we encode the **raw** translation string (`text_raw`) so the model sees natural punctuation and function words—closer to how SBERT-style models are used in practice.

The **same** `len(preprocessed_tokens) >= 3` filter keeps verse lists aligned with earlier notebooks.
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
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
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
print("PyTorch device available:", "cuda" if torch.cuda.is_available() else "cpu")
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


verse_texts: list[str] = []
verse_ids: list[str] = []
preprocessed_strings: list[str] = []

for _, row in df.iterrows():
    raw = str(row["text_raw"])
    s = preprocess_text(raw)
    toks = s.split()
    if len(toks) >= 3:
        verse_texts.append(raw)
        verse_ids.append(f"{row['Surah']}:{row['Verse']}")
        preprocessed_strings.append(s)

print("Verses (>=3 content tokens after preprocess):", len(verse_texts))
print("Example verse_id:", verse_ids[0])
print("Raw snippet:", verse_texts[0][:120], "...")
"""
    )
)

cells.append(
    md(
        r"""# Part D — Load a sentence encoder and embed verses

**First run:** the model weights download from Hugging Face (hundreds of MB). Subsequent runs use the cache.

For a **faster dry run**, lower `MAX_VERSES` (e.g. `1500`). For full-corpus behavior, set `MAX_VERSES = None`.
"""
    )
)

cells.append(
    code(
        r"""MAX_VERSES = 5000  # set to None to encode every filtered verse (slower on CPU)

subset = slice(0, MAX_VERSES) if MAX_VERSES is not None else slice(None)
texts_run = verse_texts[subset]
ids_run = verse_ids[subset]
pre_run = preprocessed_strings[subset]

print("Encoding subset size:", len(texts_run))

MODEL_NAME = "all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
st_model = SentenceTransformer(MODEL_NAME, device=device)

emb = st_model.encode(
    texts_run,
    batch_size=64,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)
print("Sentence embedding matrix shape:", emb.shape)
"""
    )
)

cells.append(
    md(
        r"""# Part E — Inspect one verse embedding (“still just a vector”)

Each verse becomes a **fixed-length** float array. Unlike sparse TF‑IDF rows, there are **no human-readable dimensions**—only directions learned by pretraining.

This mirrors the “inspect a vector” habit from earlier embedding notebooks.
"""
    )
)

cells.append(
    code(
        r"""sample_i = 0
sample_text = texts_run[sample_i]
sample_vec = emb[sample_i]

print("Verse id:", ids_run[sample_i])
print("Text (first 200 chars):\n", sample_text[:200], ("..." if len(sample_text) > 200 else ""), sep="")
print("\nEmbedding shape:", sample_vec.shape)
print("L2 norm (should be ~1.0 if normalized):", float(np.linalg.norm(sample_vec)))
print("First 12 dimensions:", np.round(sample_vec[:12], 4))
"""
    )
)

cells.append(
    md(
        r"""# Part F — TF‑IDF baseline on the **same** verses (sparse lexical space)

We fit `TfidfVectorizer` on the **preprocessed** strings for the same rows as `emb`. This mirrors NB02 but on the filtered verse list used in this notebook.
"""
    )
)

cells.append(
    code(
        r"""tfidf = TfidfVectorizer(
    lowercase=False,
    token_pattern=r"(?u)\b\w\w+\b",
    min_df=2,
    sublinear_tf=True,
)
X_tfidf = tfidf.fit_transform(pre_run)
print("TF-IDF shape (verses, features):", X_tfidf.shape)
"""
    )
)

cells.append(
    md(
        r"""# Part G — Retrieval: dense sentence embeddings vs TF‑IDF

Queries are **natural language** strings. We normalize dense query embeddings to match `normalize_embeddings=True` on verses (cosine similarity = dot product).
"""
    )
)

cells.append(
    code(
        r"""def search_dense(query: str, top_k: int = 5):
    qv = st_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = emb @ qv
    top_idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i]), ids_run[i], texts_run[i]) for i in top_idx]


def search_tfidf(query: str, top_k: int = 5):
    q_str = preprocess_text(query)
    qv = tfidf.transform([q_str])
    sims = cosine_similarity(qv, X_tfidf)[0]
    top_idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i]), ids_run[i], texts_run[i]) for i in top_idx]


def print_hits(title: str, hits: list, snip: int = 110) -> None:
    print(title)
    print("=" * 72)
    for r, (i, sc, vid, txt) in enumerate(hits, 1):
        t = (txt[:snip] + "...") if len(txt) > snip else txt
        print(f"{r}. [{sc:.4f}] {vid}  {t}")
    print()


QUERIES = [
    "kindness and compassion from the Lord",
    "punishment for those who reject the truth",
    "garden paradise reward for the righteous",
]

for q in QUERIES:
    print("\n" + "#" * 72)
    print("QUERY:", repr(q))
    hd = search_dense(q, top_k=5)
    ht = search_tfidf(q, top_k=5)
    print_hits("Dense sentence embeddings (MiniLM)", hd)
    print_hits("TF-IDF (preprocessed query)", ht)
    d_idx = {i for i, _, _, _ in hd}
    t_idx = {i for i, _, _, _ in ht}
    print("Overlap of top-5 row indices:", sorted(d_idx & t_idx))
"""
    )
)

cells.append(
    md(
        r"""# Same retrieval as **tables** (first query only)

`pandas` + `display` make it easier to compare ranks in class. (If not in Jupyter, `display` falls back to printing.)
"""
    )
)

cells.append(
    code(
        r"""def hits_to_df(hits: list) -> pd.DataFrame:
    rows = []
    for r, (_i, sc, vid, txt) in enumerate(hits, start=1):
        snip = (txt[:100] + "...") if len(txt) > 100 else txt
        rows.append({"rank": r, "score": sc, "verse_id": vid, "snippet": snip})
    return pd.DataFrame(rows)


q0 = QUERIES[0]
print("Query:", repr(q0))
display(hits_to_df(search_dense(q0, top_k=5)))
display(hits_to_df(search_tfidf(q0, top_k=5)))
"""
    )
)

cells.append(
    md(
        r"""# Part H — Similarity score distribution for one query (diagnostic)

A quick histogram of **dense cosine similarities** across all verses in the subset shows whether the query lands in a **flat** or **peaked** distribution. Flat distributions make “top‑5” less meaningful; peaks suggest clearer separation.
"""
    )
)

cells.append(
    code(
        r"""probe_q = "mercy and forgiveness from God"
qv = st_model.encode([probe_q], convert_to_numpy=True, normalize_embeddings=True)[0]
all_sims = emb @ qv

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(all_sims, bins=40, color="#5C6BC0", edgecolor="black", alpha=0.85)
ax.axvline(float(np.max(all_sims)), color="red", linestyle="--", label="max similarity")
ax.set_title(f'Dense similarities for query: "{probe_q[:48]}..."')
ax.set_xlabel("cosine similarity (normalized embeddings)")
ax.set_ylabel("count (verses)")
ax.legend()
plt.tight_layout()
plt.show()

print("Top score:", float(np.max(all_sims)), "  Median:", float(np.median(all_sims)))
"""
    )
)

cells.append(
    md(
        r"""# Part I — STS-style toy: sentence–sentence cosine (same encoder)

**Semantic Textual Similarity (STS)** benchmarks ask how well **cosine similarity** between sentence embeddings matches human judgments. We do not run the full benchmark—only a **tiny hand-built table** so you can sanity-check the geometry.

**Read carefully:** high cosine **does not** mean “theologically equivalent” or “entailed.” It only means **close in this model’s space**.
"""
    )
)

cells.append(
    code(
        r"""STS_PAIRS = [
    ("Allah is merciful and forgiving.", "God is compassionate and kind."),
    ("Allah is merciful and forgiving.", "The fire punishes the wicked."),
    ("Be kind to your parents.", "Honor your father and your mother."),
    ("Those who reject faith face consequences.", "The garden awaits the righteous."),
    ("Pray with sincerity and patience.", "The cat sat on the mat."),
]

sents_a = [a for a, _ in STS_PAIRS]
sents_b = [b for _, b in STS_PAIRS]
emb_a = st_model.encode(sents_a, convert_to_numpy=True, normalize_embeddings=True)
emb_b = st_model.encode(sents_b, convert_to_numpy=True, normalize_embeddings=True)
coses = np.sum(emb_a * emb_b, axis=1)

rows = []
for (a, b), c in zip(STS_PAIRS, coses):
    rows.append(
        {
            "cosine": float(c),
            "sentence_a": (a[:52] + "...") if len(a) > 52 else a,
            "sentence_b": (b[:52] + "...") if len(b) > 52 else b,
        }
    )

sts_df = pd.DataFrame(rows).sort_values("cosine", ascending=False)
display(sts_df)

print("Highest pair (expected: paraphrase / related). Lowest (expected: unrelated).")
"""
    )
)

cells.append(
    md(
        r"""# Part J — PCA: a cloud of verse embeddings in 2D

We take the **first `min(120, n)`** verse vectors in the current run and project with **PCA** (linear). This is only for **visual intuition**—neighborhoods will differ if you change `MAX_VERSES` or the model.
"""
    )
)

cells.append(
    code(
        r"""sample_n = min(120, emb.shape[0])
E = emb[:sample_n]
labels = ids_run[:sample_n]

pca = PCA(n_components=2, random_state=42)
xy = pca.fit_transform(E)

pca_df = pd.DataFrame({"PC1": xy[:, 0], "PC2": xy[:, 1], "verse_id": labels})

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.45, s=32, c="teal", edgecolors="k", linewidths=0.2)
step = max(1, sample_n // 18)
for i in range(0, sample_n, step):
    ax.annotate(
        pca_df.loc[i, "verse_id"],
        (pca_df.loc[i, "PC1"], pca_df.loc[i, "PC2"]),
        fontsize=6,
        alpha=0.85,
    )
ax.set_title(f"PCA of first {sample_n} verse embeddings (sentence-transformers)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.tight_layout()
plt.show()

print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
"""
    )
)

cells.append(
    md(
        r"""# Part K — Top‑10 dense hits as a bar chart (one query)

Horizontal bars make **ranking** visible at a glance (same scores as `search_dense`, different view).
"""
    )
)

cells.append(
    code(
        r"""BAR_QUERY = "kindness to parents"
top10 = search_dense(BAR_QUERY, top_k=10)
bar_df = pd.DataFrame(
    [{"verse_id": vid, "score": sc, "rank": r} for r, (_i, sc, vid, _t) in enumerate(top10, start=1)]
)
bar_df = bar_df.sort_values("score", ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=bar_df, x="score", y="verse_id", palette="Greens_r")
ax.invert_yaxis()
ax.set_title(f'Dense retrieval — top 10 for "{BAR_QUERY}"')
ax.set_xlabel("cosine similarity")
ax.set_ylabel("verse_id")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md(
        r"""# Part L — Discussion prompts, limits, and **Notebook 7**

### Discussion prompts (≈10 minutes)

1. When can dense retrieval **succeed without keyword overlap**? Give one example from your printed results.  
2. When can it still **feel wrong** even with a high score?  
3. Why is cosine similarity **not** entailment or “approval” of a verse for a query?  
4. Why does brute-force `emb @ q` become painful on **very large** corpora?

### What sentence embeddings still do **not** solve

| Limitation | Notes |
|---|---|
| **Hallucination risk if misused** | Similarity is not entailment; do not treat neighbors as “provenance.” |
| **Domain / register mismatch** | Pretraining may not match religious English; fine-tuning may be needed. |
| **Compute + memory** | Large batches and long texts cost more than TF‑IDF. |
| **Static at inference** | One vector per string; some newer systems use late interaction (ColBERT) or LLMs. |

### Limitation → next step

| What NB06 improves | What is still weak | What comes next |
|---|---|---|
| Whole-string dense similarity | Fast **nearest-neighbor** at millions+ of vectors | **NB07 — FAISS / ANN** |
| Semantic-style ranking | Grounded answers with citations | **NB08 — RAG** (in full series) |
| Query ↔ verse matching | Index maintenance, updates, filters | **Vector DB** topics |

```text
Notebook 6  →  Sentence embeddings (dense semantic retrieval)
Notebook 7  →  Vector search at scale (FAISS / ANN) — same ideas, bigger corpora
```

---

## Mini self-check

- [ ] I can explain one reason **mean-pooled Word2Vec** is weaker than a **sentence encoder** for negation / multi-word meaning.  
- [ ] I can run **dense retrieval** and **TF‑IDF retrieval** on the same query and compare overlap qualitatively.  
- [ ] I can read the **STS toy table** and point to one **high** and one **low** pair that match intuition.  
- [ ] I can explain one caveat of **PCA** for interpreting embedding clouds.  
- [ ] I can name one **deployment** concern (latency, disk, GPU) for sentence models.

When you are ready, open **Notebook 7 — Vector Search with FAISS** (or the next notebook in your local track).

---
"""
    )
)

cells.append(
    code(
        r"""# --- Optional take-home exercises (uncomment / duplicate cells) ---
# 1. Swap MODEL_NAME (e.g. "sentence-transformers/all-mpnet-base-v2") and compare STS pair ordering.
# 2. Set MAX_VERSES=None on GPU; log encode seconds and peak RAM.
# 3. Fit PCA on 500 verses but color points by Surah (parse verse_id) — any visible clusters?
# 4. Bar-plot TF-IDF top-10 for the same BAR_QUERY and compare verse_id lists to dense top-10.
# 5. Add mean-pooled FastText (NB05) verse vectors for the same subset; MRR@10 vs dense for three queries.

print("Optional exercise prompts are in the comments above this cell.")
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

out_path = Path(__file__).parent / "NLP_Workshop_06_Sentence_Embeddings_and_Semantic_Search.ipynb"
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote:", out_path)

if __name__ == "__main__":
    print("Done.")
