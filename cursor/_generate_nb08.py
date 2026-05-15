"""Emit NLP_Workshop_08_RAG_and_LLM_Retrieval.ipynb — RAG pipeline, rerank, local + API LLM, grounding check."""
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
        r"""# NLP Workshop — Notebook 8  
## RAG and LLM-based Retrieval

*Cursor assisted — this workshop track is drafted and maintained with Cursor for reproducibility.*

**Corpus:** English translations of the Quran (CSV)  
**Translation column:** `daryabadi`

---

## Where you are in the roadmap

```text
… →  NB07  FAISS / ANN  →  NB08  RAG + LLM  (end of Cursor NB01–NB08 arc)
                              ▲
                       YOU ARE HERE
```

This notebook **closes the main Cursor workshop arc (NB01–NB08)**. Further topics (eval harnesses, agents, hybrid search at scale) are extensions, not required to finish the core story.

**NB06–NB07** gave you **dense vectors** and **fast nearest-neighbor search**.

This notebook closes the loop for a **question-answering-shaped** workflow:

1. **Retrieve** candidate verses (bi-encoder / FAISS, same pattern as NB07).  
2. **Rerank** (optional but recommended): a **cross-encoder** scores **(query, passage)** pairs more accurately than a single dot product.  
3. **Augment** a **prompt** with cited snippets `[Surah:Verse]`.  
4. **Generate** with either a **small local** seq2seq model (**`flan-t5-small`**) or an **optional OpenAI** chat call if `OPENAI_API_KEY` is set.  
5. **Grounding check (toy):** extract `Surah:Verse` patterns from the answer and compare to retrieved IDs.

> This is still a **lab**, not a production assistant: no content moderation, no logging, no online eval.

**Prerequisites:** Notebooks 1–7.

---

## Learning outcomes

1. Implement a **retrieve → rerank → prompt → generate** RAG-shaped pipeline on the workshop corpus.  
2. Explain why a **cross-encoder reranker** can fix mistakes from **bi-encoder-only** top‑k.  
3. Build **chat-style** `messages` for an API LLM and compare tone to a **local** T5.  
4. Run a **toy grounding** check: verse IDs in the answer vs retrieved set.  
5. Name what a **real** deployment would still add (safety, eval, chunking, filters).  
6. Compare **no-context** vs **RAG** answers with the **same** generator (A/B in Part F).

---
"""
    )
)

cells.append(
    md(
        r"""# Part A — RAG vs “just retrieval” vs “just an LLM”

| Pattern | What the user gets | Typical failure |
|---|---|---|
| **Retriever only** | A ranked list of verses | No fluent synthesis; user must read everything. |
| **LLM only** | Fluent text | **Hallucination**; no guaranteed tie to your corpus. |
| **RAG** | Fluent text **anchored** to retrieved passages | Bad retrieval → confident wrong answers; context too long → truncation. |

**RAG** does not remove responsibility: you still verify claims against the cited passages.

---
"""
    )
)

cells.append(
    md(
        r"""# Part B — Setup: data, bi-encoder, FAISS `IndexFlatIP`

Same CSV discovery and **`>=3` token** verse filter as NB06–NB07. We encode **`MAX_VERSES`** verses and build an **exact** inner-product index (L2‑normalized rows → cosine ranking).

**Downloads:** `sentence-transformers` + FAISS are local. **`cross-encoder/...`** (Part D) and **`flan-t5-small`** (Part F) download on first use.
"""
    )
)

cells.append(
    code(
        r"""from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import nltk
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

sns.set_theme(style="whitegrid", context="notebook")
print("PyTorch:", torch.__version__, "  CUDA:", torch.cuda.is_available())
print("OK: core imports loaded.")
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
    if len(s.split()) >= 3:
        verse_texts.append(raw)
        verse_ids.append(f"{row['Surah']}:{row['Verse']}")

print("Filtered verses:", len(verse_texts))
"""
    )
)

cells.append(
    code(
        r"""MAX_VERSES = 3000

texts_run = verse_texts[:MAX_VERSES]
ids_run = verse_ids[:MAX_VERSES]
N = len(texts_run)

BI_ENCODER = "all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
st_model = SentenceTransformer(BI_ENCODER, device=device)

emb = st_model.encode(
    texts_run,
    batch_size=64,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(np.float32)
emb = np.ascontiguousarray(emb)
d = emb.shape[1]

index_flat = faiss.IndexFlatIP(d)
index_flat.add(emb)
print("N =", N, " dim =", d, "  FAISS ntotal:", index_flat.ntotal)
"""
    )
)

cells.append(
    md(
        r"""# Part C — Bi-encoder retrieve (pool → rerank)

We pull **`POOL_K`** neighbors with FAISS, then keep the top **`FINAL_K`** after reranking (next part). A larger pool gives the cross-encoder more to work with.
"""
    )
)

cells.append(
    code(
        r"""POOL_K = 12
FINAL_K = 4


def encode_query_vec(q: str) -> np.ndarray:
    v = st_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
    return np.ascontiguousarray(v.astype(np.float32).reshape(1, d))


def retrieve_biencoder(query: str, k: int = POOL_K):
    qv = encode_query_vec(query)
    scores, idx = index_flat.search(qv, k)
    out = []
    for j, sc in zip(idx[0], scores[0]):
        j = int(j)
        out.append((ids_run[j], texts_run[j], float(sc)))
    return out


USER_QUERY = "What does the text say about kindness toward parents?"
pool = retrieve_biencoder(USER_QUERY, k=POOL_K)

print("Query:", repr(USER_QUERY))
print("Pool size:", len(pool), "  top bi-encoder score:", pool[0][2])
"""
    )
)

cells.append(
    md(
        r"""# Part D — Cross-encoder reranking (`sentence_transformers.CrossEncoder`)

A **cross-encoder** jointly encodes **query + passage** and outputs a **relevance score**. It is slower than a dot product, but often **reorders** bi-encoder mistakes when used on a **small pool**.

We use **`cross-encoder/ms-marco-MiniLM-L-6-v2`** (downloads on first run).
"""
    )
)

cells.append(
    code(
        r"""from sentence_transformers import CrossEncoder

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ce_model = CrossEncoder(RERANKER, device=device)


def rerank_pool(query: str, pool: list[tuple[str, str, float]], topn: int = FINAL_K):
    pairs = [[query, txt] for _vid, txt, _ in pool]
    xs = np.asarray(ce_model.predict(pairs, show_progress_bar=False), dtype=np.float64)
    order = np.argsort(-xs)[:topn]
    reranked = [pool[int(i)] for i in order]
    return reranked, xs[order]


reranked_hits, rerank_scores = rerank_pool(USER_QUERY, pool, topn=FINAL_K)

print("After rerank (top", FINAL_K, "):")
for r, ((vid, txt, _bio), rr) in enumerate(zip(reranked_hits, rerank_scores), 1):
    snip = txt.replace("\n", " ")[:140]
    print(f"  {r}. [{float(rr):.4f} rerank] {vid}  {snip}...")

plot_df = pd.DataFrame(
    {"verse_id": [p[0] for p in reranked_hits], "cross_encoder_score": [float(s) for s in rerank_scores]}
)
fig, ax = plt.subplots(figsize=(7, 3.2))
sns.barplot(data=plot_df, x="cross_encoder_score", y="verse_id", palette="Blues_r")
ax.set_title("Cross-encoder scores (reranked finalists)")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md(
        r"""# Part E — Grounded **RAG prompt** (citations + strict instructions)

We format context blocks as **`[Surah:Verse] text...`** and ask the model to **cite verse IDs** when it uses a claim. We also ask for a final line **`SOURCES:`** listing IDs used—this supports the toy parser in Part I.
"""
    )
)

cells.append(
    code(
        r"""def build_rag_prompt(query: str, retrieved: list[tuple[str, str, float]], max_chars: int = 2000) -> str:
    blocks = []
    for vid, txt, _sc in retrieved:
        t = txt.replace("\n", " ").strip()
        if len(t) > 420:
            t = t[:420] + "..."
        blocks.append(f"[{vid}] {t}")
    ctx = "\n\n".join(blocks)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n\n[... context truncated ...]"

    return (
        "Rules:\n"
        "- Use ONLY the Context passages below.\n"
        "- If the Context does not answer the question, say you cannot tell from the Context.\n"
        "- When you state a factual claim, cite the verse ID like [12:34] right after the claim.\n"
        "- End your answer with a line exactly: SOURCES: id1, id2, ... (comma-separated Surah:Verse IDs you used).\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"
    )


rag_prompt = build_rag_prompt(USER_QUERY, reranked_hits)
print("--- Prompt preview (first 1000 chars) ---\n")
print(rag_prompt[:1000])
print("\n... total chars:", len(rag_prompt))
"""
    )
)

cells.append(
    md(
        r"""# Part F — A/B: **no context** vs **RAG** (same local model)

The most important teaching contrast: **same question**, **same generator**, but **with vs without** retrieved passages in the prompt.

- **No-context:** the model can only use **parametric memory** (weights from pretraining)—often vague or wrong for corpus-specific questions.  
- **RAG:** the model sees **verbatim** `[Surah:Verse]` snippets—answers should be more **corpus-tied** (if the model obeys instructions).

Set **`USE_LOCAL_LLM = False`** to skip downloads; you still have prompts and retrieval tables.
"""
    )
)

cells.append(
    code(
        r"""USE_LOCAL_LLM = True
GEN_LOCAL = "google/flan-t5-small"

answer_nocontext: str | None = None
answer_local: str | None = None

nocontext_prompt = (
    "Answer the question in one short paragraph. "
    "You do NOT have access to any external text or scripture passages.\n\n"
    f"Question: {USER_QUERY}\n\nAnswer:"
)


def generate_flan(prompt: str, tok, mdl, dev: str, max_new_tokens: int = 120) -> str:
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(dev)
    with torch.no_grad():
        out_ids = mdl.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=3)
    return tok.decode(out_ids[0], skip_special_tokens=True).strip()


if not USE_LOCAL_LLM:
    print("USE_LOCAL_LLM is False — skipping A/B generation (inspect rag_prompt / nocontext_prompt).")
else:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as e:
        print("pip install transformers", e)
    else:
        tok = AutoTokenizer.from_pretrained(GEN_LOCAL)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(GEN_LOCAL)
        mdl.eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(dev)

        print("=" * 72)
        print("A) NO CONTEXT (parametric memory only)")
        print("=" * 72)
        answer_nocontext = generate_flan(nocontext_prompt, tok, mdl, dev)
        print(answer_nocontext)

        print("\n" + "=" * 72)
        print("B) RAG (retrieved + reranked passages in prompt)")
        print("=" * 72)
        answer_local = generate_flan(rag_prompt, tok, mdl, dev)
        print(answer_local)
"""
    )
)

cells.append(
    md(
        r"""# Part G — Multi-query retrieval + rerank (batch table)

Same pipeline for **three** user questions. Compare bi-encoder **#1** vs reranker **#1**—they often disagree when the query is paraphrased.
"""
    )
)

cells.append(
    code(
        r"""BATCH_QUERIES = [
    "What does the text say about kindness toward parents?",
    "How is mercy of God described?",
    "Reward for those who do good deeds",
]

rows = []
for q in BATCH_QUERIES:
    p = retrieve_biencoder(q, k=POOL_K)
    rh, _ = rerank_pool(q, p, topn=FINAL_K)
    rows.append(
        {
            "query": (q[:50] + "...") if len(q) > 50 else q,
            "bi_enc_top1": p[0][0],
            "rerank_top1": rh[0][0],
            "agree_top1": p[0][0] == rh[0][0],
        }
    )

display(pd.DataFrame(rows))
"""
    )
)

cells.append(
    md(
        r"""# Part H — **Optional API LLM** (OpenAI Chat Completions)

If **`OPENAI_API_KEY`** is set in the environment (or in a `.env` loaded by your tooling), we send the **same** `rag_prompt` as a **user** message with a short **system** instruction.

Install: **`pip install openai`**. If the import fails or the key is missing, this cell prints a skip message.

**Do not commit API keys.** Use environment variables or secret managers only.
"""
    )
)

cells.append(
    code(
        r"""OPENAI_MODEL = "gpt-4o-mini"
answer_api: str | None = None

if not os.getenv("OPENAI_API_KEY"):
    print("OPENAI_API_KEY not set — skipping API generation.")
else:
    try:
        from openai import OpenAI
    except ImportError:
        print("pip install openai  (package not found)")
    else:
        client = OpenAI()
        messages = [
            {
                "role": "system",
                "content": "You answer using only the Context inside the user message. "
                "Cite verse IDs like [2:83]. End with SOURCES: id1, id2.",
            },
            {"role": "user", "content": rag_prompt},
        ]
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=280,
            temperature=0.2,
        )
        answer_api = (resp.choices[0].message.content or "").strip()
        print(f"--- {OPENAI_MODEL} ---\n")
        print(answer_api)
"""
    )
)

cells.append(
    md(
        r"""# Part I — Toy **grounding** check: `SOURCES:` vs retrieved IDs

We **parse** `Surah:Verse` tokens from the model output and compare to the **FINAL_K** reranked passages. This is **not** a full entailment model—just a **string hygiene** check for teaching.

If the model forgets the **`SOURCES:`** line, the parser returns an empty set (also informative).
"""
    )
)

cells.append(
    code(
        r"""VID_PAT = re.compile(r"\b\d{1,3}:\d{1,3}\b")


def verse_ids_in_text(s: str) -> set[str]:
    return set(VID_PAT.findall(s or ""))


retrieved_ids = {vid for vid, _t, _s in reranked_hits}

for label, ans in [("local_rag", answer_local), ("openai_rag", answer_api)]:
    if not ans:
        continue
    cited = verse_ids_in_text(ans)
    tail = ans.split("SOURCES:", 1)
    sources_line = tail[1].strip() if len(tail) > 1 else ""
    declared = verse_ids_in_text(sources_line)

    print("===", label, "===")
    print("All verse-like tokens in full answer:", sorted(cited))
    print("Declared on SOURCES line:", sorted(declared))
    print("Overlap with retrieved pool:", sorted(declared & retrieved_ids))
    print("Declared but NOT in retrieved:", sorted(declared - retrieved_ids))
    print()
"""
    )
)

cells.append(
    md(
        r"""# Part J — What this notebook is **not** (teaching prototype)

| Missing piece | Why it matters | Direction (beyond NB08) |
|---|---|---|
| **Hybrid retrieval** | Dense-only can miss exact keywords | BM25 + dense + **RRF** fusion |
| **Faithfulness / NLI check** | Model may ignore context or overclaim | NLI verifier, RAGAS-style metrics |
| **Strong grounding audit** | Regex on `SOURCES:` is a toy | Attribution models + human rubrics |
| **Safety / policy layer** | Prompt injection, toxic I/O | Moderation APIs, allowlists, logging |
| **Eval harness** | No NDCG / answer quality over time | Labeled Q&A set, regression tests |
| **Multi-turn chat** | Single question only | Conversation memory, query rewrite |
| **Production indexing** | `IndexFlatIP` on thousands of verses is a demo | **IVF / HNSW** (NB07), sharding, filters |
| **Chunking strategy** | Whole verses may be long or noisy | Overlapping chunks + parent `verse_id` |

We **do** include **cross-encoder reranking** and an optional **API LLM**—unlike minimal demos that stop at retrieval-only.

---
"""
    )
)

cells.append(
    md(
        r"""# Part K — Discussion prompts and **production** next steps

### Discussion prompts (≈10 minutes)

1. When did **reranking** change the top‑1 verse vs bi-encoder alone (Part G table)? Why might that matter for user trust?  
2. What failure mode does **`SOURCES:`** parsing catch, and what does it **still miss**?  
3. Why is **temperature 0.2** a reasonable default for factual-ish RAG answers? When might you increase it?  
4. If you deployed this publicly, what **policy** checks would you add before showing generated text?

4. If you deployed this publicly, what **policy** checks would you add before showing generated text?  
5. In Part F, how did the **no-context** answer differ from **RAG** for the same query?

### Limitation → next step

| What NB08 demonstrates | What a production track adds |
|---|---|
| RAG-shaped pipeline | **Eval harness**, **logging**, **A/B tests** |
| Optional API call | **Rate limits**, **cost dashboards**, **fallback models** |
| Toy grounding string check | **NLI / attribution models**, human rubrics |

---

## Mini self-check

- [ ] I can draw **retrieve → rerank → prompt → generate** as a flowchart.  
- [ ] I can explain **one** reason a cross-encoder reranker can beat bi-encoder top‑1.  
- [ ] I can describe how **`OPENAI_API_KEY`** changes the notebook behavior without editing code.  
- [ ] I can interpret **`Declared but NOT in retrieved`** from Part I.  
- [ ] I can explain **one** difference between **no-context** and **RAG** generation (Part F).

---

### Optional exercises (comments only — next cell)

---
"""
    )
)

cells.append(
    md(
        r"""# Part L — **Workshop summary** (NB01 → NB08)

This notebook **finishes** the core Cursor track. Each step fixed a limitation of the previous representation.

```text
Keyword / exact match
   →  Bag of Words (counts)
   →  TF-IDF (weighted sparse)
   →  N-grams (local word order)
   →  Word2Vec (dense words)
   →  FastText (subwords / OOV)
   →  Sentence embeddings (dense whole text)
   →  FAISS (fast vector lookup)
   →  RAG (retrieve + cite + generate)
```

| NB | Topic | Key idea |
|---|---|---|
| **01** | Preprocess + BoW | Text → sparse word counts; exact match limits |
| **02** | TF-IDF | Down-weight common terms; ranked lexical retrieval |
| **03** | N-grams | Short phrases as features; stopword trap |
| **04** | Word2Vec | Dense word vectors; mean-pooled verse search |
| **05** | FastText | Character n-grams; OOV / morphology |
| **06** | Sentence embeddings | One vector per verse; dense vs TF-IDF |
| **07** | FAISS | Exact + approximate ANN; persist index |
| **08** | RAG + LLM | Rerank → grounded prompt → generate + toy cite check |

### What you should be able to do now

- Build a **retrieval** stack on a fixed translation column with consistent preprocessing.  
- Explain when **sparse**, **word-level**, **sentence-level**, and **generative** tools each help.  
- Prototype **RAG** with citations—and name what you would add before any public deployment.

**Extension:** **Notebook 9 — Advanced RAG Techniques** (`NLP_Workshop_09_Advanced_RAG_Techniques.ipynb`) upgrades retrieval with **hybrid RRF**, **chunking**, **compression**, and **eval proxies**.

---
"""
    )
)

cells.append(
    code(
        r"""# 1. Add BM25 (e.g. rank_bm25) hybrid: merge BM25 top-20 with FAISS top-20, dedupe, then cross-encode rerank.
# 2. Split long verses into 2–3 overlapping chunks; track (verse_id, chunk_idx) in SOURCES.
# 3. Call OpenAI with response_format JSON schema: {answer, sources: [...], confidence}.
# 4. Build a 20-question CSV; batch-run retrieval+rerank; manually label "good/bad" top-1 for a mini leaderboard.

print("NB08 RAG+LLM lab cells complete.")
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

out_path = Path(__file__).parent / "NLP_Workshop_08_RAG_and_LLM_Retrieval.ipynb"
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote:", out_path)

if __name__ == "__main__":
    print("Done.")
