# NLP Workshop: From Bag-of-Words to Advanced RAG

**A complete, hands-on NLP workshop series — built for students, usable by anyone.**

This workshop teaches Natural Language Processing the way it actually evolved — each technique motivated by the failure of the previous one. By the end, you will have built a working Retrieval-Augmented Generation pipeline entirely from scratch, running on your own machine with no GPU and no API key.

---

## The Learning Arc

Every notebook critiques the one before it.

<div class="grid cards" markdown>

-   **NB01 — Bag-of-Words**

    Count words. Build a vocabulary. Make your first search engine.

    *Limitation: all words treated equally.*

-   **NB02 — TF-IDF**

    Weight words by how rare they are across the corpus.

    *Limitation: still ignores word meaning.*

-   **NB03 — N-Grams**

    Capture word sequences and local context.

    *Limitation: misses long-range semantics.*

-   **NB04 — Word2Vec**

    Dense vectors that encode meaning. Words close in meaning sit close in space.

    *Limitation: one vector per word, nothing for sentences.*

-   **NB05 — FastText**

    Subword embeddings that handle morphology and out-of-vocabulary words.

    *Limitation: still word-level, not sentence-level.*

-   **NB06 — Sentence Embeddings**

    One vector per sentence using SBERT. Semantic search that understands paraphrases.

    *Limitation: brute-force search is too slow at scale.*

-   **NB07 — FAISS**

    Index millions of vectors for fast approximate nearest-neighbor search.

    *Limitation: search returns passages, not answers.*

-   **NB08 — RAG**

    Retrieve relevant passages, inject them into a prompt, generate a grounded answer.

    *Limitation: naive retrieval, no re-ranking, no evaluation.*

-   **NB09 — Advanced RAG**

    Hybrid retrieval, cross-encoder re-ranking, HyDE, chunking, RAGAS evaluation.

    *The complete production-grade pipeline.*

</div>

---

## Quick Start

```bash
git clone https://github.com/haqnawaz99/BOW-to-Embeddings.git
cd BOW-to-Embeddings
pip install -r requirements.txt
jupyter notebook Claude/NB01_Text_Preprocessing_and_BoW.ipynb
```

All notebooks are in the `Claude/` folder. Run them in order, starting from NB01.

---

## Bring Your Own Data

The workshop ships with a sample multi-column text CSV so you can run everything immediately. But the entire pipeline is **dataset-agnostic** — swap in any corpus you care about:

- News articles
- Legal documents
- Research papers
- Product reviews
- Company knowledge bases

One line to change in each notebook. Everything else stays the same.

---

## What You Will Build

By the end of NB09 you will have assembled this pipeline from scratch:

```
Your text corpus
        |
        v
[Preprocessing]  tokenize, clean, stopword removal
        |
        v
[BM25 index] + [SBERT FAISS index]
        |               |
        +---> RRF <-----+    Hybrid retrieval
               |
               v
   [Cross-encoder re-ranker]   Precision boost
               |
               v
      [Prompt builder]         Augmentation with citations
               |
               v
   [Local LLM — flan-t5-small] Generation
               |
               v
    Grounded answer with source citations
```

---

## Requirements

| Requirement | Detail |
|---|---|
| Python | 3.10 or higher |
| GPU | Not required — all notebooks run on CPU |
| API key | Not required — local models only |
| Internet | Required for first-run model downloads (~500 MB total) |
| OS | Windows, macOS, Linux |

---

## About This Series

These 9 notebooks were developed iteratively using Claude (Anthropic), cross-evaluated against parallel implementations built with Codex and Cursor AI, and continuously improved by merging the best ideas from all three passes.

Each notebook includes:

- Detailed markdown explanations for every section
- Fully executable code cells
- Generated visualizations
- 5 reflection questions with detailed answers
