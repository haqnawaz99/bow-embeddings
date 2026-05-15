"""Emit NLP_Workshop_09_Advanced_RAG_Techniques.ipynb — hybrid RRF, rerank, chunks, compression, eval proxy."""
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
        r"""# NLP Workshop — Notebook 9  
## Advanced RAG Techniques

*Cursor assisted — extension beyond the NB01–NB08 core arc.*

**Corpus:** English translations of the Quran (CSV)  
**Translation column:** `daryabadi`

---

## Where you are in the roadmap

```text
NB01–NB08  Core workshop (BoW → … → FAISS → baseline RAG)
      ↓
NB09  Advanced RAG   ← YOU ARE HERE
```

**Notebook 8** gave you a working **retrieve → rerank → prompt → generate** loop. In practice, that **naive** stack often fails because of **retrieval** and **context assembly**, not only because the LLM is weak.

This notebook upgrades the **retrieval layer** with techniques common in production RAG:

| NB08 baseline | NB09 upgrade |
|---|---|
| Dense-only (bi-encoder) | **Hybrid:** BM25 + dense fused with **RRF** |
| Verse-level units | **Sentence chunks** with overlap (finer precision) |
| Single query embedding | **HyDE** + **multi-query** dense recall |
| Full passage in prompt | **Contextual compression** (top sentences per chunk) |
| Toy `SOURCES:` parse | **Faithfulness / relevance** proxy scores |
| One pipeline | **Naive vs advanced** comparison on the same queries |

**Prerequisites:** Notebooks 6–8 (embeddings, FAISS, RAG). Local **`flan-t5-small`** optional for HyDE-style expansion.

---
"""
    )
)

cells.append(
    md(
        r"""# Part A — Why naive RAG breaks (diagnose before you upgrade)

| Failure | Symptom | NB09 lever |
|---|---|---|
| **Lexical mismatch** | Query says `kindness`, verse says `benevolence` | Dense + **BM25** hybrid |
| **Wrong granularity** | Long verse mixes two topics | **Chunking** |
| **Recall gap** | Right verse not in top‑k | **HyDE**, **multi-query**, larger pool before rerank |
| **Ranking noise** | Right verse in pool but not top‑4 | **Cross-encoder rerank** |
| **Prompt noise** | Irrelevant sentences distract the LLM | **Compression** |
| **No quality signal** | “Looks fine” in demo | **Proxy metrics** |

---
"""
    )
)

cells.append(
    md(
        r"""# Part B — Setup: data, chunks, BM25 corpus, dense index

We reuse the workshop CSV and **`>=3` token** verse filter. Then we build **retrieval units** as **sentence chunks** (not only full verses).

Tune **`MAX_VERSES`** for speed; chunking multiplies index size.
"""
    )
)

cells.append(
    code(
        r"""from __future__ import annotations

import re
import warnings
from collections import defaultdict

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

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


def tokenize_for_bm25(text: str) -> list[str]:
    return preprocess_text(text).split()


verse_texts: list[str] = []
verse_ids: list[str] = []
surah_nums: list[int] = []

for _, row in df.iterrows():
    raw = str(row["text_raw"])
    s = preprocess_text(raw)
    if len(s.split()) >= 3:
        verse_texts.append(raw)
        verse_ids.append(f"{row['Surah']}:{row['Verse']}")
        surah_nums.append(int(row["Surah"]))

print("Filtered verses:", len(verse_texts))
"""
    )
)

cells.append(
    code(
        r"""MAX_VERSES = 2500
CHUNK_SENTS = 2
CHUNK_OVERLAP = 1

texts_v = verse_texts[:MAX_VERSES]
ids_v = verse_ids[:MAX_VERSES]
surahs_v = surah_nums[:MAX_VERSES]


def make_sentence_chunks(text: str, verse_id: str, surah: int, chunk_sents: int = CHUNK_SENTS, overlap: int = CHUNK_OVERLAP):
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    if not sents:
        return []
    if len(sents) <= chunk_sents:
        return [{"chunk_id": f"{verse_id}#0", "verse_id": verse_id, "surah": surah, "text": text}]
    out = []
    step = max(1, chunk_sents - overlap)
    for start in range(0, len(sents), step):
        piece = " ".join(sents[start : start + chunk_sents])
        if not piece:
            continue
        cid = f"{verse_id}#{start}"
        out.append({"chunk_id": cid, "verse_id": verse_id, "surah": surah, "text": piece})
        if start + chunk_sents >= len(sents):
            break
    return out


chunks: list[dict] = []
for raw, vid, sur in zip(texts_v, ids_v, surahs_v):
    chunks.extend(make_sentence_chunks(raw, vid, sur))

chunk_df = pd.DataFrame(chunks)
print("Chunks:", len(chunk_df), "  (from", len(texts_v), "verses)")
print(chunk_df.head(3))
"""
    )
)

cells.append(
    code(
        r"""# BM25 on preprocessed chunk tokens
chunk_texts = chunk_df["text"].tolist()
bm25_corpus = [tokenize_for_bm25(t) for t in chunk_texts]
bm25 = BM25Okapi(bm25_corpus)

# Dense index on raw chunk text (readable for LLM)
BI_ENCODER = "all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
st_model = SentenceTransformer(BI_ENCODER, device=device)

emb_chunks = st_model.encode(
    chunk_texts,
    batch_size=64,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).astype(np.float32)
emb_chunks = np.ascontiguousarray(emb_chunks)
d = emb_chunks.shape[1]

index_chunks = faiss.IndexFlatIP(d)
index_chunks.add(emb_chunks)
print("FAISS chunks:", index_chunks.ntotal, "  dim:", d)
"""
    )
)

cells.append(
    md(
        r"""# Part C — Reciprocal Rank Fusion (RRF): BM25 + dense

**RRF** merges ranked lists without normalizing incompatible scores:

\[
\text{RRF}(d) = \sum_{\text{lists } L} \frac{1}{k + \text{rank}_L(d)}
\]

We use **`k = 60`** (common default). Each chunk is keyed by **`chunk_id`**.
"""
    )
)

cells.append(
    code(
        r"""RRF_K = 60
POOL_K = 25


def dense_rank_chunk_ids(query: str, k: int = POOL_K) -> list[str]:
    qv = st_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    qv = np.ascontiguousarray(qv.astype(np.float32).reshape(1, d))
    _, idx = index_chunks.search(qv, k)
    return [chunk_df.iloc[int(j)]["chunk_id"] for j in idx[0]]


def bm25_rank_chunk_ids(query: str, k: int = POOL_K) -> list[str]:
    qtoks = tokenize_for_bm25(query)
    scores = bm25.get_scores(qtoks)
    order = np.argsort(-scores)[:k]
    return [chunk_df.iloc[int(i)]["chunk_id"] for i in order]


def rrf_fuse(rank_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for rlist in rank_lists:
        for rank, cid in enumerate(rlist):
            scores[cid] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def rows_for_chunk_ids(cids: list[str]) -> pd.DataFrame:
  rows = []
  lookup = chunk_df.set_index("chunk_id")
  for cid in cids:
      if cid in lookup.index:
          rows.append(lookup.loc[cid])
  return pd.DataFrame(rows)


DEMO_Q = "kindness and duty toward parents"
dense_ids = dense_rank_chunk_ids(DEMO_Q)
bm25_ids = bm25_rank_chunk_ids(DEMO_Q)
fused = rrf_fuse([bm25_ids, dense_ids])[:8]

print("Top RRF chunk_ids:")
for cid, sc in fused:
    row = chunk_df[chunk_df.chunk_id == cid].iloc[0]
    print(f"  {sc:.4f}  {cid}  {row['text'][:90]}...")
"""
    )
)

cells.append(
    md(
        r"""# Part D — Cross-encoder rerank (precision stage)

**Recall:** hybrid pool (`POOL_K`). **Precision:** `cross-encoder/ms-marco-MiniLM-L-6-v2` scores **(query, chunk)** jointly and keeps **`FINAL_K`**.
"""
    )
)

cells.append(
    code(
        r"""RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FINAL_K = 4
ce_model = CrossEncoder(RERANKER, device=device)


def advanced_retrieve(query: str, pool_k: int = POOL_K, final_k: int = FINAL_K) -> pd.DataFrame:
    d_ids = dense_rank_chunk_ids(query, k=pool_k)
    b_ids = bm25_rank_chunk_ids(query, k=pool_k)
    fused_ids = [cid for cid, _ in rrf_fuse([b_ids, d_ids])[:pool_k]]
    pool_df = rows_for_chunk_ids(fused_ids)
    if pool_df.empty:
        return pool_df
    pairs = [[query, t] for t in pool_df["text"].tolist()]
    xs = np.asarray(ce_model.predict(pairs, show_progress_bar=False), dtype=np.float64)
    pool_df = pool_df.copy()
    pool_df["rerank_score"] = xs
    pool_df = pool_df.sort_values("rerank_score", ascending=False).head(final_k)
    return pool_df.reset_index(drop=True)


adv_hits = advanced_retrieve(DEMO_Q)
print("Advanced pipeline top", FINAL_K, "chunks:")
display(adv_hits[["chunk_id", "verse_id", "rerank_score", "text"]])
"""
    )
)

cells.append(
    md(
        r"""# Part E — HyDE: Hypothetical Document Embeddings

**HyDE** (Gao et al., 2022): for a **short query**, ask an LLM to write a **hypothetical passage** that might answer it, then **embed that passage** instead of the raw query.

**Why it can help:** the hypothetical text is longer and may use vocabulary closer to the corpus (e.g. `mercy`, `compassion`, `parents`).

**Why it can fail:** the LLM may hallucinate content that does not exist in your corpus—always validate retrieved chunks.

We use **`google/flan-t5-small`** (same family as NB08). Set **`USE_HYDE = False`** to skip generation and compare only raw-query dense ranks.
"""
    )
)

cells.append(
    code(
        r"""USE_HYDE = True
GEN_HYDE = "google/flan-t5-small"


def dense_rank_from_text(text: str, k: int = POOL_K) -> list[str]:
    qv = st_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    qv = np.ascontiguousarray(qv.astype(np.float32).reshape(1, d))
    _, idx = index_chunks.search(qv, k)
    return [chunk_df.iloc[int(j)]["chunk_id"] for j in idx[0]]


def hyde_hypothetical(query: str) -> str:
    prompt = (
        "Write one short paragraph in formal English, as if from a religious translation, "
        f"that could answer this question: {query}"
    )
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(GEN_HYDE)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(GEN_HYDE).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev)
    inputs = tok(prompt, return_to_tensors="pt", truncation=True, max_length=256).to(dev)
    with torch.no_grad():
        out_ids = mdl.generate(**inputs, max_new_tokens=80, do_sample=False, num_beams=2)
    return tok.decode(out_ids[0], skip_special_tokens=True).strip()


if USE_HYDE:
    hypo = hyde_hypothetical(DEMO_Q)
    print("HyDE hypothetical passage:\n", hypo, "\n", sep="")
    raw_top = dense_rank_chunk_ids(DEMO_Q)[:5]
    hyde_top = dense_rank_from_text(hypo)[:5]
    print("Dense top-5 chunk_ids — raw query:     ", raw_top)
    print("Dense top-5 chunk_ids — HyDE passage:", hyde_top)
    print("Overlap:", len(set(raw_top) & set(hyde_top)), "/ 5")
else:
    print("USE_HYDE is False — enable to compare raw-query vs HyDE dense retrieval.")
"""
    )
)

cells.append(
    md(
        r"""# Part F — Multi-query dense recall (paraphrase variants)

We retrieve with **three query strings**, union chunk IDs, then still use **RRF + rerank** on the merged pool. This improves **recall** when wording differs from the corpus.
"""
    )
)

cells.append(
    code(
        r"""def multi_query_variants(user_q: str) -> list[str]:
    variants = [
        user_q,
        user_q.replace("kindness", "goodness").replace("parents", "mother and father"),
        "duty compassion mercy " + user_q,
    ]
    return list(dict.fromkeys(v.strip() for v in variants if v.strip()))


def advanced_retrieve_multi(query: str, pool_k: int = POOL_K, final_k: int = FINAL_K) -> pd.DataFrame:
    variants = list(dict.fromkeys(multi_query_variants(query)))[:3]
    dense_lists = [dense_rank_chunk_ids(v, k=pool_k) for v in variants]
    bm25_lists = [bm25_rank_chunk_ids(query, k=pool_k)]
    fused_ids = [cid for cid, _ in rrf_fuse(bm25_lists + dense_lists)[:pool_k]]
    pool_df = rows_for_chunk_ids(fused_ids)
    pairs = [[query, t] for t in pool_df["text"].tolist()]
    xs = np.asarray(ce_model.predict(pairs, show_progress_bar=False), dtype=np.float64)
    pool_df = pool_df.copy()
    pool_df["rerank_score"] = xs
    return pool_df.sort_values("rerank_score", ascending=False).head(final_k).reset_index(drop=True)


mq_hits = advanced_retrieve_multi(DEMO_Q)
print("Multi-query advanced top chunks:")
display(mq_hits[["chunk_id", "verse_id", "rerank_score"]])
"""
    )
)

cells.append(
    md(
        r"""# Part G — Contextual compression (sentence-level)

Even a good chunk may contain **off-topic sentences**. We score each sentence with the **same cross-encoder** and keep the top **`KEEP_SENTS`** per chunk before building the RAG prompt.
"""
    )
)

cells.append(
    code(
        r"""KEEP_SENTS = 2


def compress_chunks(query: str, hits_df: pd.DataFrame, keep_sents: int = KEEP_SENTS) -> list[dict]:
    compressed = []
    for _, row in hits_df.iterrows():
        sents = [s.strip() for s in sent_tokenize(str(row["text"])) if s.strip()]
        if len(sents) <= keep_sents:
            kept = sents
        else:
            pairs = [[query, s] for s in sents]
            sc = np.asarray(ce_model.predict(pairs, show_progress_bar=False), dtype=np.float64)
            order = np.argsort(-sc)[:keep_sents]
            kept = [sents[int(i)] for i in sorted(order)]
        compressed.append(
            {
                "chunk_id": row["chunk_id"],
                "verse_id": row["verse_id"],
                "text": " ".join(kept),
            }
        )
    return compressed


compressed = compress_chunks(DEMO_Q, adv_hits)
for c in compressed:
    print(f"[{c['verse_id']}] {c['text'][:120]}...")
"""
    )
)

cells.append(
    md(
        r"""# Part H — Naive vs advanced: same queries, different top‑1

| Pipeline | Retrieval |
|---|---|
| **Naive** | Dense bi-encoder on **full verses** (NB08 style) |
| **Advanced** | Chunk hybrid RRF + cross-encoder rerank + compression |
"""
    )
)

cells.append(
    code(
        r"""# Naive verse-level dense index (for comparison only)
emb_verses = st_model.encode(
    texts_v,
    batch_size=64,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(np.float32)
emb_verses = np.ascontiguousarray(emb_verses)
index_verses = faiss.IndexFlatIP(d)
index_verses.add(emb_verses)


def naive_dense_verse_top1(query: str) -> str:
    qv = st_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    qv = np.ascontiguousarray(qv.astype(np.float32).reshape(1, d))
    _, idx = index_verses.search(qv, 1)
    return ids_v[int(idx[0][0])]


COMPARE_QS = [
    "kindness toward parents",
    "mercy and forgiveness from God",
    "reward in paradise for the righteous",
]

rows = []
for q in COMPARE_QS:
    naive_vid = naive_dense_verse_top1(q)
    adv_vid = advanced_retrieve(q).iloc[0]["verse_id"]
    rows.append({"query": q, "naive_top1_verse": naive_vid, "advanced_top1_verse": adv_vid, "agree": naive_vid == adv_vid})

display(pd.DataFrame(rows))
"""
    )
)

cells.append(
    md(
        r"""# Part I — Proxy evaluation: faithfulness + answer relevance

Full **RAGAS** uses LLM judges. Here we use **cheap proxies** on a generated answer:

- **Faithfulness (proxy):** fraction of **content words** in the answer that appear in the compressed context (very coarse).  
- **Relevance (proxy):** cross-encoder score **(query, answer)**.

Optional: generate with **`flan-t5-small`** on the compressed prompt (same pattern as NB08).
"""
    )
)

cells.append(
    code(
        r"""def build_advanced_prompt(query: str, compressed_chunks: list[dict], max_chars: int = 1800) -> str:
    blocks = [f"[{c['verse_id']}] {c['text']}" for c in compressed_chunks]
    ctx = "\n\n".join(blocks)[:max_chars]
    return (
        "Use ONLY the context. Cite verse IDs. End with SOURCES: id1, id2.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"
    )


USE_LOCAL_LLM = False
GEN_LOCAL = "google/flan-t5-small"

prompt_adv = build_advanced_prompt(DEMO_Q, compressed)
answer_adv: str | None = None

if USE_LOCAL_LLM:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(GEN_LOCAL)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(GEN_LOCAL).eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(dev)
        inputs = tok(prompt_adv, return_tensors="pt", truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            out_ids = mdl.generate(**inputs, max_new_tokens=100, do_sample=False, num_beams=2)
        answer_adv = tok.decode(out_ids[0], skip_special_tokens=True).strip()
        print("Generated answer:\n", answer_adv)
    except Exception as e:
        print("Generation skipped:", e)
else:
    print("USE_LOCAL_LLM=False — using a stub answer for metric demo.")
    answer_adv = (
        "The context speaks of kindness and duty toward parents [example]. "
        "SOURCES: (demo)"
    )

ctx_blob = " ".join(c["text"] for c in compressed).lower()
ans_words = set(re.findall(r"[a-z]{4,}", (answer_adv or "").lower()))
ctx_words = set(re.findall(r"[a-z]{4,}", ctx_blob))
faithfulness_proxy = len(ans_words & ctx_words) / max(1, len(ans_words))
relevance_proxy = float(ce_model.predict([[DEMO_Q, answer_adv]])[0])

print(f"Faithfulness proxy (word overlap): {faithfulness_proxy:.3f}")
print(f"Relevance proxy (CE query-answer): {relevance_proxy:.3f}")
"""
    )
)

cells.append(
    md(
        r"""# Part J — Metadata filter (Surah-scoped retrieval)

Production systems filter by **language**, **collection**, or **ACL**. Here we restrict candidates to **`allowed_surahs`** before reranking.
"""
    )
)

cells.append(
    code(
        r"""def advanced_retrieve_surah(query: str, allowed_surahs: list[int], pool_k: int = 40) -> pd.DataFrame:
    d_ids = dense_rank_chunk_ids(query, k=pool_k)
    b_ids = bm25_rank_chunk_ids(query, k=pool_k)
    fused_ids = [cid for cid, _ in rrf_fuse([b_ids, d_ids])]
    pool_df = rows_for_chunk_ids(fused_ids)
    pool_df = pool_df[pool_df["surah"].isin(allowed_surahs)]
    if pool_df.empty:
        return pool_df
    pairs = [[query, t] for t in pool_df["text"].tolist()]
    xs = np.asarray(ce_model.predict(pairs, show_progress_bar=False), dtype=np.float64)
    pool_df = pool_df.copy()
    pool_df["rerank_score"] = xs
    return pool_df.sort_values("rerank_score", ascending=False).head(FINAL_K).reset_index(drop=True)


print("Surah filter demo (Surah 1–2 only):")
display(advanced_retrieve_surah("guidance for believers", allowed_surahs=[1, 2])[["verse_id", "surah", "rerank_score"]])
"""
    )
)

cells.append(
    md(
        r"""# Part K — Discussion, limits, workshop closure

### Discussion prompts

1. When did **BM25** beat dense alone on a query—and when did it hurt?  
2. Why is **RRF** safer than adding raw BM25 and cosine scores?  
3. What breaks if **chunk overlap** is zero?  
4. Why is the **faithfulness proxy** here *not* enough for production?

### What is still missing

| Gap | Production direction |
|---|---|
| **Learned fusion** | Train rerankers / LTR on click logs |
| **Multi-hop** | Iterative retrieve for complex questions |
| **Real RAGAS / human eval** | Labeled Q&A + auditor rubrics |

### Technique map (NB09)

| Technique | What it fixes |
|---|---|
| Hybrid + RRF | Lexical + semantic recall |
| Cross-encoder rerank | Precision in top‑k |
| HyDE | Short-query → richer dense query vector |
| Multi-query | Wording / recall gaps |
| Chunking + overlap | Granularity + boundary loss |
| Compression | Prompt noise |
| Proxy metrics | Eyeballing at scale |

---

## Mini self-check

- [ ] I can explain **RRF** in one sentence.  
- [ ] I can name **recall vs precision** stages in this notebook.  
- [ ] I can contrast **naive verse dense** vs **advanced chunk** top‑1.  
- [ ] I can explain **HyDE** and one way it can mislead retrieval.  
- [ ] I know why proxy metrics are **not** a full eval suite.

---
"""
    )
)

cells.append(
    code(
        r"""# 1. Fuse HyDE dense ranks into RRF (add a third list to rrf_fuse).
# 2. Tune RRF_K (1 vs 60 vs 500) and plot top-1 churn on COMPARE_QS.
# 3. Log precision@4 on a hand-labeled sheet of 15 queries.

print("NB09 Advanced RAG complete.")
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

out_path = Path(__file__).parent / "NLP_Workshop_09_Advanced_RAG_Techniques.ipynb"
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote:", out_path)

if __name__ == "__main__":
    print("Done.")
