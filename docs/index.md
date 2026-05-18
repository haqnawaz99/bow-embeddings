# NLP Workshop: From Bag-of-Words to Advanced RAG

**A complete, hands-on NLP series — built for students, usable by anyone.**

Learn Natural Language Processing the way it actually evolved. Every technique is motivated by the failure of the previous one. By the end you will have built a working Retrieval-Augmented Generation pipeline entirely from scratch — no GPU, no API key, no cloud account.

---

## The Learning Arc

```
Raw Text
   |
   v
[Bag-of-Words]         Count words. Ignore meaning.
   |  fails: equal weights
   v
[TF-IDF]               Weight by rarity. Better ranking.
   |  fails: no semantics
   v
[N-Grams]              Capture word sequences.
   |  fails: vocabulary explosion
   v
[Word2Vec]             Dense vectors. Meaning from context.
   |  fails: one vector per word, no sentences
   v
[FastText]             Subword embeddings. Handle OOV.
   |  fails: still word-level
   v
[Sentence Embeddings]  One vector per sentence. Semantic search.
   |  fails: brute-force search too slow at scale
   v
[FAISS]                Index millions of vectors. Fast ANN search.
   |  fails: returns passages, not answers
   v
[RAG]                  Retrieve + Augment + Generate.
   |  fails: naive retrieval, no re-ranking, no evaluation
   v
[Advanced RAG]         Hybrid search, re-ranking, HyDE, RAGAS eval.
```

---

## Notebooks

| # | Topic | Key technique | What it fixes |
|---|---|---|---|
| [NB01](notebooks/nb01.md) | Text Preprocessing & Bag-of-Words | CountVectorizer, cosine similarity | Starting point |
| [NB02](notebooks/nb02.md) | TF-IDF and Ranked Retrieval | TfidfVectorizer, IDF weighting | Equal word weights |
| [NB03](notebooks/nb03.md) | N-Grams and Context | ngram_range, bigram LM | Word order blindness |
| [NB04](notebooks/nb04.md) | Word2Vec Embeddings | CBOW, Skip-Gram, PCA, t-SNE | Sparse, no semantics |
| [NB05](notebooks/nb05.md) | FastText and Subword Embeddings | Character n-grams, OOV handling | Unknown words |
| [NB06](notebooks/nb06.md) | Sentence Embeddings | SBERT all-MiniLM-L6-v2 | Word-level only |
| [NB07](notebooks/nb07.md) | Vector Databases and FAISS | IndexFlatIP, IndexIVFFlat, recall@K | Brute-force search |
| [NB08](notebooks/nb08.md) | RAG and LLM Retrieval | Retrieve-Augment-Generate, flan-t5 | Passages not answers |
| [NB09](notebooks/nb09.md) | Advanced RAG | BM25+SBERT, cross-encoder, HyDE | Naive RAG limitations |

---

## Quick Start

```bash
git clone https://github.com/haqnawaz99/bow-embeddings.git
cd bow-embeddings
pip install -r requirements.txt
jupyter notebook Claude/NB01_Text_Preprocessing_and_BoW.ipynb
```

Run notebooks in order from NB01. Each one builds on the previous.

---

## Concepts

New to NLP? Start with the concept articles before diving into the notebooks:

- [What is NLP?](concepts/what-is-nlp.md) — definition, applications, why language is hard for machines
- [Text Preprocessing](concepts/text-preprocessing.md) — tokenization, stopwords, stemming explained
- [Sparse vs Dense Representations](concepts/sparse-vs-dense.md) — the leap from BoW to embeddings
- [What are Embeddings?](concepts/embeddings.md) — vectors, meaning, semantic space
- [What is RAG?](concepts/what-is-rag.md) — retrieve, augment, generate explained simply

---

## Bring Your Own Data

The workshop ships with a sample multi-column text CSV. The pipeline is **dataset-agnostic** — swap in any corpus:

```python
# One line to change in the config cell of any notebook
TEXT_COLUMN = "your_column_name"
```

News articles, legal documents, research papers, product reviews — the techniques are the same.

---

## Requirements

| Item | Detail |
|---|---|
| Python | 3.10 or higher |
| GPU | Not required — all notebooks run on CPU |
| API key | Not required — local models only |
| First-run downloads | ~500 MB (models cached after first run) |

---

## About This Series

9 notebooks developed iteratively with Claude (Anthropic), cross-evaluated against parallel implementations built with Codex and Cursor AI, and continuously improved by merging the best ideas from all three. Each notebook includes detailed theory, executable code, generated visualisations, and 5 reflection questions with answers.
