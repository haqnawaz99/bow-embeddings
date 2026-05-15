"""Emit NLP_Workshop_08_RAG_and_LLM_Retrieval.ipynb — retrieve, cite, generate (starter RAG lab)."""
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
## RAG and LLM-based Retrieval (starter)

*Cursor assisted — this workshop track is drafted and maintained with Cursor for reproducibility.*

**Corpus:** English translations of the Quran (CSV)  
**Translation column:** `daryabadi`

---

## Where you are in the roadmap

```text
… →  NB07  FAISS / ANN  →  NB08  RAG + LLM  →  (extensions: eval, safety, agents)
                              ▲
                       YOU ARE HERE
```

**Retrieval** (NB06–NB07) answers: *“Which passages are near this query in embedding space?”*

**Retrieval-Augmented Generation (RAG)** adds: *“Draft a helpful response **conditioned on** those passages,”* usually with **citations** so readers can verify claims against sources.

This notebook is a **minimal** RAG-shaped pipeline:

1. **Dense retrieve** top passages with the same sentence encoder + **FAISS `IndexFlatIP`** pattern as NB07.  
2. **Build a single prompt** that includes **verbatim snippets** tagged with **`Surah:Verse`**.  
3. Optionally run a **small local** `google/flan-t5-small` generator (downloads on first use). **No cloud API keys** are required by default.

> Production RAG adds reranking, citations formatting, guardrails, logging, and evaluation. This lab is intentionally small.

**Prerequisites:** Notebooks 1–7 (especially NB06–NB07).

---
"""
    )
)

cells.append(
    md(
        r"""# Part A — What “RAG” means in one pass

## Ingredients

| Stage | Role |
|---|---|
| **Retriever** | Find candidate text chunks (here: full verses) likely relevant to the user query. |
| **Augment** | Paste those chunks into a **prompt context** (with IDs so they are citable). |
| **Generate** | Ask an LLM to produce an answer **using** the context (ideally with quotes / verse IDs). |

## Failure modes you should teach out loud

- **Hallucination:** the model invents facts not supported by the context.  
- **Lost-in-the-middle / truncation:** the model never sees important evidence if the prompt is too long.  
- **Wrong retrieval:** garbage context in → confident garbage out.

---
"""
    )
)

cells.append(
    md(
        r"""# Part B — Imports, verses, embeddings, FAISS index

Same **CSV discovery** and **`>=3` token** verse filter as NB06–NB07. We encode a **subset** for speed, build **`IndexFlatIP`**, and reuse it for retrieval below.

Set `USE_LOCAL_LLM = True` in Part E only if you accept a **~300MB** `flan-t5-small` download the first time.
"""
    )
)

cells.append(
    code(
        r"""from __future__ import annotations

import re
import warnings
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

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
        r"""MAX_VERSES = 2500  # raise if you have GPU / time

texts_run = verse_texts[:MAX_VERSES]
ids_run = verse_ids[:MAX_VERSES]
N = len(texts_run)
print("Encoding N =", N)

MODEL_NAME = "all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
st_model = SentenceTransformer(MODEL_NAME, device=device)

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
print("FAISS IndexFlatIP ntotal:", index_flat.ntotal, "  dim:", d)
"""
    )
)

cells.append(
    md(
        r"""# Part C — Retrieve: query → top‑k verses

We return **`(verse_id, text, inner_product_score)`** tuples. Scores are **cosine-like** because vectors are L2‑normalized (same convention as NB06–NB07).
"""
    )
)

cells.append(
    code(
        r"""def encode_query_vec(q: str) -> np.ndarray:
    v = st_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
    return np.ascontiguousarray(v.astype(np.float32).reshape(1, d))


def retrieve_passages(query: str, k: int = 4):
    qv = encode_query_vec(query)
    scores, idx = index_flat.search(qv, k)
    out = []
    for j, sc in zip(idx[0], scores[0]):
        j = int(j)
        out.append((ids_run[j], texts_run[j], float(sc)))
    return out


USER_QUERY = "What does the text say about kindness toward parents?"
hits = retrieve_passages(USER_QUERY, k=4)

print("Query:", repr(USER_QUERY))
for rank, (vid, txt, sc) in enumerate(hits, 1):
    snip = txt.replace("\n", " ")[:180]
    print(f"\n{rank}. [{sc:.4f}] {vid}\n   {snip}...")
"""
    )
)

cells.append(
    md(
        r"""# Part D — Build a **grounded prompt** (citations in the context string)

We explicitly wrap each snippet with its **`Surah:Verse`** id so the model (and the human reader) can tie claims back to evidence.

We keep the context **short** so a small T5 model fits its **512-token** budget after truncation.
"""
    )
)

cells.append(
    code(
        r"""def build_rag_prompt(query: str, retrieved: list[tuple[str, str, float]], max_chars: int = 2200) -> str:
    blocks = []
    for vid, txt, _sc in retrieved:
        t = txt.replace("\n", " ").strip()
        if len(t) > 450:
            t = t[:450] + "..."
        blocks.append(f"[{vid}] {t}")
    ctx = "\n\n".join(blocks)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n\n[... context truncated ...]"

    return (
        "You are a careful assistant. Use ONLY the context below. "
        "If the context does not answer the question, say you cannot tell from the context. "
        "Mention verse IDs when you cite a claim.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"
    )


rag_prompt = build_rag_prompt(USER_QUERY, hits)
print("--- Prompt preview (first 900 chars) ---\n")
print(rag_prompt[:900])
print("\n... [total chars]", len(rag_prompt))
"""
    )
)

cells.append(
    md(
        r"""# Part E — Optional: **small local** generator (`flan-t5-small`)

**Default:** `USE_LOCAL_LLM = True` runs a **seq2seq** model on CPU or GPU. First run downloads weights.

If your environment blocks downloads or runs out of RAM, set **`USE_LOCAL_LLM = False`**: you still have retrieval + prompt inspection (the core teaching).

**Cloud APIs (OpenAI, etc.):** not used here; paste your own API pattern in the optional exercises cell if you teach that separately.
"""
    )
)

cells.append(
    code(
        r"""USE_LOCAL_LLM = True
GEN_MODEL = "google/flan-t5-small"

if not USE_LOCAL_LLM:
    print("USE_LOCAL_LLM is False — skipping generation. Review rag_prompt above.")
else:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as e:
        print("Install transformers: pip install transformers", e)
    else:
        tok = AutoTokenizer.from_pretrained(GEN_MODEL)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
        mdl.eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(dev)

        inputs = tok(rag_prompt, return_tensors="pt", truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            out_ids = mdl.generate(**inputs, max_new_tokens=96, do_sample=False, num_beams=2)
        answer = tok.decode(out_ids[0], skip_special_tokens=True).strip()

        print("\n--- Model answer ---\n")
        print(answer)
"""
    )
)

cells.append(
    md(
        r"""# Part F — What this notebook is **not** yet

| Gap | Why it matters later |
|---|---|
| **No reranker** | Cross-encoders or ColBERT-style reranking often beat single-vector top‑k. |
| **No answer grounding check** | You should verify claims against cited spans (model can still drift). |
| **No eval harness** | NDCG, faithfulness metrics, human rubrics—essential for real products. |
| **No safety / policy layer** | Production systems filter inputs/outputs and log abuse patterns. |

---

## Mini self-check

- [ ] I can name the **three** letters in **RAG** and what each stage does.  
- [ ] I can explain one way **bad retrieval** breaks RAG even if the LLM is strong.  
- [ ] I can read the **prompt** and point to where **citations** enter the context.

---

### Limitation → next step (beyond this course draft)

| What NB08 introduces | What a full course adds |
|---|---|
| Prompt + citations | **Evaluation**, **guardrails**, **tool use** |
| One-shot generation | **Chunking**, **multi-hop**, **query rewriting** |
| Local tiny model | **Hosted LLMs**, **cost/latency** tradeoffs |

---

### Optional exercises (comments only — next cell)

---
"""
    )
)

cells.append(
    code(
        r"""# 1. Swap retriever to top-8 chunks; observe if the small model loses focus or invents more.
# 2. Add a second stage: ask the model to output ONLY verse IDs it used, then parse and validate.
# 3. Wire an OpenAI/Anthropic call (API key in .env) using the same rag_prompt — compare tone vs flan-t5-small.
# 4. Log retrieval scores + final answer to a CSV for a batch of 20 queries (mini offline eval).

print("NB08 starter complete. Extend with eval + safety before any public deployment.")
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
