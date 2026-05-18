# What is Retrieval-Augmented Generation?

**Retrieval-Augmented Generation (RAG)** is a technique that connects a search system to a language model. Instead of asking a model to answer from memory alone, RAG first retrieves relevant passages from a knowledge base, then asks the model to answer using those passages as context.

---

## The Problem RAG Solves

Large Language Models (LLMs) like GPT-4 or flan-t5 have impressive knowledge baked into their parameters. But they have three serious problems when used alone:

**1. Hallucination**
Models generate fluent, confident text — but it may be factually wrong. They cannot easily say "I don't know".

**2. Stale knowledge**
Models are trained on data up to a cutoff date. They cannot answer questions about recent events without retraining.

**3. No citations**
Models cannot tell you *where* their answer came from, making verification impossible.

RAG addresses all three by grounding the model in retrieved evidence.

---

## The Three Stages

```
User Query
    |
    v
[ 1. RETRIEVE ]
  Encode the query as a vector.
  Search a vector index for the most relevant passages.
  Return top-k passages with scores.
    |
    v  top-k passages
[ 2. AUGMENT ]
  Inject passages into a structured prompt.
  Add citation tags [Surah:Verse].
  Add grounding instructions: "use ONLY the context below".
    |
    v  grounded prompt
[ 3. GENERATE ]
  Pass prompt to an LLM.
  Model reads context and generates an answer.
  Answer cites specific passages.
    |
    v
  Grounded answer with citations
```

---

## A Concrete Example

**Query:** "What does the text say about patience in hardship?"

**Step 1 — Retrieve:**
The query is encoded as a 384-dimensional vector using SBERT. FAISS searches the index and returns:
```
[2:155] "And certainly, We shall test you with something of fear,
         hunger, loss of wealth, lives and fruits..."  score=0.73

[39:10] "Only those who are patient shall receive their reward
         in full, without reckoning."  score=0.68
```

**Step 2 — Augment:**
```
You are a careful assistant. Use ONLY the context below.
Cite verse IDs when you make a claim.

--- Context ---
[2:155] And certainly, We shall test you with something of fear...
[39:10] Only those who are patient shall receive their reward...

--- Question ---
What does the text say about patience in hardship?

--- Answer ---
```

**Step 3 — Generate:**
```
According to [2:155], difficulty is a test that includes
fear, hunger, and loss. [39:10] promises that those who
remain patient will receive their full reward.
```

---

## RAG vs. Bare LLM

| Property | Bare LLM | RAG |
|---|---|---|
| Knowledge source | Parametric (weights) | Retrieved documents |
| Hallucination risk | High | Reduced — grounded |
| Citability | None | Passage IDs cited |
| Updatable | Requires retraining | Swap the index |
| Transparency | Black box | Retrievals are visible |

---

## The Retrieval Stack

RAG quality depends heavily on retrieval quality. This workshop builds up the full stack:

| Component | Technique | Notebook |
|---|---|---|
| Keyword retrieval | BM25 | NB09 |
| Semantic retrieval | SBERT + FAISS | NB06–NB07 |
| Hybrid fusion | BM25 + SBERT + RRF | NB09 |
| Re-ranking | Cross-encoder | NB09 |
| Query expansion | HyDE | NB09 |
| Chunking | Sentence-boundary overlap | NB09 |
| Evaluation | RAGAS-style faithfulness | NB09 |

---

## Advanced RAG

The basic RAG pipeline (NB08) has known weaknesses:

- Dense-only retrieval misses exact-match terms
- No re-ranking — top-k by cosine may be off-topic
- Full verses passed to LLM — includes irrelevant sentences
- No way to measure if the answer is actually grounded

NB09 fixes all of these with advanced techniques.

---

→ Start building: [NB08 — RAG and LLM-based Retrieval](../notebooks/nb08.md)
