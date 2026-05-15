"""Emit NLP_Workshop_07_FAISS_and_Vector_Search.ipynb — FAISS index, IVF recall, timing vs brute force."""
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
        r"""# NLP Workshop — Notebook 7  
## FAISS and Vector Search

*Cursor assisted — this workshop track is drafted and maintained with Cursor for reproducibility.*

**Corpus:** English translations of the Quran (CSV)  
**Translation column:** `daryabadi`

---

## Where you are in the roadmap

```text
… →  NB06  Sentence embeddings  →  NB07  FAISS / ANN  →  NB08+  RAG / LLM retrieval
                                              ▲
                                       YOU ARE HERE
```

**Notebook 6** encoded each verse as a **dense vector** and ranked by **cosine similarity** (often implemented as a matrix–vector product `emb @ q` when vectors are **L2‑normalized**).

That **exact** brute-force scan is \(O(N \cdot d)\) per query. It is fine for a few thousand verses on CPU, but it does not scale to **millions** of passages without **indexes**.

**FAISS** (Facebook AI Similarity Search) is a library for **approximate nearest neighbor (ANN)** search and related **exact** structures. This notebook builds:

1. An **exact** **inner-product** index (`IndexFlatIP`) — equivalent ranking to `emb @ q` for normalized vectors.  
2. A small **IVFFlat** index — trains coarse clusters for faster **approximate** search; we measure **recall** vs brute force.  
3. A simple **timing** comparison (still toy-scale; the lesson is qualitative).

**Prerequisites:** Notebooks 1–6 (especially dense vectors in NB06).

---

## Learning outcomes

1. Explain why **brute-force** dense retrieval hits a **latency wall** as \(N\) grows.  
2. Build a **FAISS** `IndexFlatIP` on **float32** L2‑normalized verse embeddings and retrieve **top‑k**.  
3. Verify **exact** index results match **numpy** brute ranking (up to ties).  
4. Train a small **`IndexIVFFlat`** (inner product), tune **`nprobe`**, and report **recall@K** vs brute top‑K.  
5. Compare **wall-clock** of brute matmul vs `index.search` at this scale (interpret cautiously).  
6. Describe what **Notebook 8 (RAG)** adds beyond “return similar verses.”  
7. Explain why teams still use **vector databases** alongside libraries like FAISS, and how **persisted** indexes help.

---
"""
    )
)

cells.append(
    md(
        r"""# Part A — From “matrix multiply” to “index object”

## Brute force (NB06 pattern)

For normalized rows \(x_i\) and query \(q\):

\[
\text{cosine}(x_i, q) = x_i^\top q
\]

So **top‑k** retrieval is “compute all dot products, pick the largest.”

## What an index buys you

- **Same math, better packaging:** `IndexFlatIP` stores vectors and runs optimized **batch** inner-product search (BLAS / threading).  
- **Approximation:** `IndexIVFFlat` partitions the space into **Voronoi cells** (inverted lists). At query time you visit only a few lists (`nprobe`) → often **much faster**, sometimes **misses** true top‑k neighbors.

## Engineering vocabulary

| Term | One-line meaning |
|---|---|
| **Flat** | No compression / no clustering shortcut — typically **exact** for the metric used. |
| **IVF** | *Inverted file* — cluster centroids route the query to a subset of vectors. |
| **`nprobe`** | How many clusters to visit (quality vs speed). |

---
"""
    )
)

cells.append(
    md(
        r"""# Part B — Imports, data, sentence embeddings (same stack as NB06)

We reload the CSV, apply the **same** verse filter (`>=3` content tokens after `preprocess_text`), and encode **`MAX_VERSES`** raw verses with **`all-MiniLM-L6-v2`**.

> Set `MAX_VERSES = None` for the full filtered corpus if your machine can handle it.
"""
    )
)

cells.append(
    code(
        r"""from __future__ import annotations

import re
import time
import warnings
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import nltk
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

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

try:
    faiss.omp_set_num_threads(min(8, faiss.omp_get_max_threads()))
except Exception:
    pass

sns.set_theme(style="whitegrid", context="notebook")
try:
    print("FAISS omp max threads:", faiss.omp_get_max_threads())
except Exception:
    print("FAISS omp max threads: (unknown)")
print("PyTorch device:", "cuda" if torch.cuda.is_available() else "cpu")
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

for _, row in df.iterrows():
    raw = str(row["text_raw"])
    s = preprocess_text(raw)
    toks = s.split()
    if len(toks) >= 3:
        verse_texts.append(raw)
        verse_ids.append(f"{row['Surah']}:{row['Verse']}")

print("Filtered verses:", len(verse_texts))
"""
    )
)

cells.append(
    code(
        r"""MAX_VERSES = 5000  # None = all filtered verses

sl = slice(0, MAX_VERSES) if MAX_VERSES is not None else slice(None)
texts_run = verse_texts[sl]
ids_run = verse_ids[sl]
N = len(texts_run)
print("Encoding N =", N, "verses")

MODEL_NAME = "all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
st_model = SentenceTransformer(MODEL_NAME, device=device)

emb = st_model.encode(
    texts_run,
    batch_size=64,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).astype(np.float32)

emb = np.ascontiguousarray(emb)
d = emb.shape[1]
print("emb dtype/shape:", emb.dtype, emb.shape, "  L2 row norms (mean):", float(np.mean(np.linalg.norm(emb, axis=1))))
"""
    )
)

cells.append(
    md(
        r"""# Part C — `IndexFlatIP` (exact inner product = cosine, when normalized)

`IndexFlatIP` stores all vectors and returns **exact** top‑inner-product neighbors. For **L2‑normalized** vectors, maximizing the dot product is the same ranking rule as **cosine similarity** in NB06.
"""
    )
)

cells.append(
    code(
        r"""index_flat = faiss.IndexFlatIP(d)
index_flat.add(emb)
print("IndexFlatIP ntotal:", index_flat.ntotal)
"""
    )
)

cells.append(
    md(
        r"""# Part D — Sanity check: FAISS vs brute-force `argsort`

We encode a query, run **`index_flat.search`**, and compare indices to **`emb @ q`** ranking.
"""
    )
)

cells.append(
    code(
        r"""QUERY = "kindness and compassion toward parents"


def encode_query(q: str) -> np.ndarray:
    v = st_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
    return np.ascontiguousarray(v.astype(np.float32).reshape(1, d))


def brute_topk(X: np.ndarray, qrow: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sims = (X @ qrow.T).ravel()
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


q = encode_query(QUERY)
k = 10
bf_idx, bf_scores = brute_topk(emb, q, k)
D_flat, I_flat = index_flat.search(q, k)

ok = np.array_equal(bf_idx, I_flat[0])
print("Query:", repr(QUERY))
print("Brute indices:", bf_idx.tolist())
print("FAISS indices:", I_flat[0].tolist())
print("Indices match exactly:", ok)
print("Score max abs diff:", float(np.max(np.abs(bf_scores - D_flat[0]))))

for rank, (i, sc) in enumerate(zip(I_flat[0], D_flat[0]), 1):
    vid = ids_run[int(i)]
    snip = texts_run[int(i)][:100].replace("\n", " ")
    print(f"  {rank:2d}. [{sc:.4f}] {vid}  {snip}...")
"""
    )
)

cells.append(
    md(
        r"""# Part E — `IndexIVFFlat` (approximate) + **recall@K** vs brute

We train an **IVFFlat** index with **`METRIC_INNER_PRODUCT`**. Training runs **k-means** on the embedding vectors to build `nlist` centroids.

**Recall@K (diagnostic):** fraction of the brute-force **top‑K** IDs that still appear inside the approximate **top‑K′** results (here \(K = K' = 50\)). It is **not** a formal ANN benchmark—just a classroom sanity metric.
"""
    )
)

cells.append(
    code(
        r"""nlist = min(64, max(8, N // 40))
print("IVF nlist:", nlist)

quantizer = faiss.IndexFlatIP(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
if not index_ivf.is_trained:
    index_ivf.train(emb)
index_ivf.add(emb)
index_ivf.nprobe = min(16, nlist)
print("IVFFlat trained, ntotal:", index_ivf.ntotal, "  nprobe:", index_ivf.nprobe)

K = 50
bf_top, _ = brute_topk(emb, q, K)
_, ivf_top = index_ivf.search(q, K)
hit = len(set(bf_top.tolist()) & set(ivf_top[0].tolist()))
recall = hit / K
print(f"Recall@{K} (overlap of brute vs IVF top-{K}):", recall)

for nprobe_try in (1, 4, 16, min(32, nlist)):
    index_ivf.nprobe = nprobe_try
    _, ivf_top2 = index_ivf.search(q, K)
    hit2 = len(set(bf_top.tolist()) & set(ivf_top2[0].tolist()))
    print(f"  nprobe={nprobe_try:2d}  recall@{K} = {hit2 / K:.3f}")
"""
    )
)

cells.append(
    md(
        r"""# Part F — Timing sketch (warm up, then average)

Microbenchmarks are **noisy** (CPU frequency, BLAS, other processes). Treat numbers as **order-of-magnitude intuition**, not competitive benchmarks.

We repeat one query many times: **numpy dot** vs **`IndexFlatIP.search`** vs **`IndexIVFFlat.search`**.
"""
    )
)

cells.append(
    code(
        r"""REPS = 80
warm = 5

queries = [
    "mercy and forgiveness",
    "punishment of the wicked",
    "garden and paradise for believers",
]


def bench(name: str, fn, reps: int = REPS) -> float:
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def run_brute():
    for qq in queries:
        v = encode_query(qq)
        _ = brute_topk(emb, v, 10)


def run_flat():
    for qq in queries:
        v = encode_query(qq)
        _ = index_flat.search(v, 10)


def run_ivf():
    for qq in queries:
        v = encode_query(qq)
        _ = index_ivf.search(v, 10)


t_brute = bench("brute", run_brute)
t_flat = bench("flat", run_flat)
t_ivf = bench("ivf", run_ivf)

print(f"Mean time over {REPS} reps (3 queries each rep), after warm={warm}:")
print(f"  brute np.dot pipeline: {t_brute*1000:.3f} ms / rep")
print(f"  IndexFlatIP.search:    {t_flat*1000:.3f} ms / rep")
print(f"  IndexIVFFlat.search: {t_ivf*1000:.3f} ms / rep  (nprobe={index_ivf.nprobe})")
print("\nAt toy N, gaps may be small; the point is the *API* and when IVF wins at large N.")
"""
    )
)

cells.append(
    md(
        r"""# Part G — FAISS vs **vector databases** (vocabulary)

**FAISS** is a **library** for similarity search: vectors in, neighbors out. It does **not** replace a full **document store** (versioning, access control, joins, rich filtering on metadata).

**Vector databases** (commercial or open products) typically combine:

- **Storage** for vectors + **metadata** (surah, language, ACL, timestamps)  
- **Filtered ANN** (“nearest among rows where …”)  
- **Operations** teams care about: backups, replication, multi-tenant quotas

**Common pattern:** PostgreSQL / Elasticsearch / cloud search for **metadata + text**, plus a **vector index** (sometimes FAISS under the hood) for **ANN**.

For this course, FAISS teaches the **core ANN mechanics**; products package those mechanics for production.

---
"""
    )
)

cells.append(
    md(
        r"""# Part H — Persist the index: **`write_index` / `read_index`**

In production you separate **(1) building embeddings** from **(2) serving queries**. Saving the FAISS object avoids recomputing the index every process restart.

We write **`IndexFlatIP`** to a small folder under the repo (gitignored in real life; here it is a teaching artifact).
"""
    )
)

cells.append(
    code(
        r"""INDEX_DIR = Path("faiss_workshop_cache")
INDEX_DIR.mkdir(exist_ok=True)
flat_path = INDEX_DIR / "nb07_index_flat_ip.bin"

faiss.write_index(index_flat, str(flat_path))
print("Wrote:", flat_path.resolve(), "  bytes:", flat_path.stat().st_size)

index_loaded = faiss.read_index(str(flat_path))
print("Loaded ntotal:", index_loaded.ntotal, " (matches flat):", index_loaded.ntotal == index_flat.ntotal)

D_ld, I_ld = index_loaded.search(q, 5)
print("Top-5 IDs from loaded index:", I_ld[0].tolist())
print("Matches in-memory flat:", np.array_equal(I_ld[0], I_flat[0]))
"""
    )
)

cells.append(
    md(
        r"""# Part I — Discussion prompts, limits, and **Notebook 8**

### Discussion prompts (≈10 minutes)

1. When does **`IndexFlatIP`** already win over raw Python loops—even if IVF is not needed yet?  
2. Why can **IVF recall** stay high for one query but fail on another (think cluster boundaries)?  
3. What metadata might you **filter** on for Quran apps (language, surah range, licensed translation)?  
4. Why is “**vector search**” still not the same as “**correct answer**” for a user question?

### What FAISS does **not** replace

| Gap | Notes |
|---|---|
| **Embeddings quality** | FAISS only finds neighbors in the **given** vectors; garbage in → garbage out. |
| **Updates & metadata** | Production systems pair vectors with **filters** (language, surah, ACL). FAISS is not a full document DB. |
| **Approximation risk** | IVF / HNSW / PQ can **miss** true top‑k; tune **recall–latency** with `nprobe`, `efSearch`, etc. |
| **Answers, not just neighbors** | Returning verses is retrieval; **RAG** adds generation + citations (**NB08**). |

### Limitation → next step

| What NB07 improves | What is still weak | What comes next |
|---|---|---|
| Fast ANN patterns (flat / IVF) | Grounded **answers** with citations | **NB08 — RAG** |
| Persisted index files | Safety, policy, and UI around answers | **NB08+** |
| Vector math at scale | Product-grade filtering + ops | **Vector DB** products |

---

## Mini self-check

- [ ] I can explain why **`IndexFlatIP`** matches brute-force ranking for L2‑normalized vectors.  
- [ ] I can explain **`nlist`** and **`nprobe`** in one sentence each.  
- [ ] I can name one reason **IVF recall** might be `< 1.0`.  
- [ ] I can explain one difference between **FAISS** and a **vector database**.  
- [ ] I can explain why **`write_index`** matters for deployment.

When you are ready, open **Notebook 8 — RAG and LLM-based Retrieval** (`NLP_Workshop_08_RAG_and_LLM_Retrieval.ipynb`).

---
"""
    )
)

cells.append(
    code(
        r"""# --- Optional take-home exercises (uncomment / duplicate cells) ---
# 1. Try IndexHNSWFlat (METRIC_INNER_PRODUCT) and compare recall@50 vs IVF at similar latency.
# 2. Set MAX_VERSES=None and replot timing; note when IVF clearly wins over flat.
# 3. Attach verse_id with IndexIDMap2 + IndexFlatIP — retrieve IDs without a parallel array.
# 4. Quantize with IndexPQ for memory savings; measure recall drop vs flat.

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

out_path = Path(__file__).parent / "NLP_Workshop_07_FAISS_and_Vector_Search.ipynb"
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote:", out_path)

if __name__ == "__main__":
    print("Done.")
