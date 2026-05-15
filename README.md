# NLP Workshop: From Bag-of-Words to Advanced RAG

A complete, hands-on NLP workshop series that teaches the history and practice of Natural Language Processing — from counting words to grounding a language model in a real corpus.

**Dataset:** Quran translations (`daryabadi` column, 6,236 verses)  
**Language:** Python 3.10+  
**Approach:** Learn each technique by building it, observe its limitation, then fix it in the next notebook.

---

## Quick Start

```powershell
# 1. Activate the virtual environment
cd "D:\Haq Nawaz\Teaching\NLP\BOW-to-Embeddings"
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (first time only)
pip install -r requirements.txt

# 3. Open the first notebook
jupyter notebook Claude/NB01_Text_Preprocessing_and_BoW.ipynb
```

> **Recommended series:** `Claude/` — the most complete and polished set, with detailed explanations, self-evaluations against peer implementations, and all known bugs fixed.

---

## The Learning Path

Each notebook addresses a limitation of the previous one.

| # | Notebook | Key Concept | Limitation it fixes |
|---|---|---|---|
| 01 | [Text Preprocessing and BoW](./Claude/NB01_Text_Preprocessing_and_BoW.ipynb) | Tokenization, stopwords, Bag-of-Words | Starting point |
| 02 | [TF-IDF and Ranked Retrieval](./Claude/NB02_TF_IDF_and_Ranked_Retrieval.ipynb) | TF-IDF weighting, cosine similarity | BoW treats all words equally |
| 03 | [N-Grams and Context](./Claude/NB03_NGrams_and_Context.ipynb) | Bigrams, language models | TF-IDF ignores word order |
| 04 | [Word2Vec Embeddings](./Claude/NB04_Word2Vec_Embeddings.ipynb) | CBOW, Skip-Gram, PCA/t-SNE | N-grams miss long-range meaning |
| 05 | [FastText and Subword Embeddings](./Claude/NB05_FastText_and_Subword_Embeddings.ipynb) | Character n-grams, OOV handling | Word2Vec fails on unknown words |
| 06 | [Sentence Embeddings and Semantic Search](./Claude/NB06_Sentence_Embeddings_and_Semantic_Search.ipynb) | SBERT, all-MiniLM-L6-v2 | Word vectors cannot represent sentences |
| 07 | [Vector Databases and FAISS](./Claude/NB07_Vector_Databases_and_FAISS.ipynb) | IndexFlatIP, IndexIVFFlat, recall@K | Brute-force search is too slow at scale |
| 08 | [RAG and LLM-based Retrieval](./Claude/NB08_RAG_and_LLM_Retrieval.ipynb) | Retrieve → Augment → Generate, flan-t5-small | Search returns passages, not answers |
| 09 | [Advanced RAG Techniques](./Claude/NB09_Advanced_RAG_Techniques.ipynb) | Hybrid BM25+SBERT, re-ranking, HyDE, chunking, RAGAS | Naive RAG has no re-ranking or evaluation |

**Total:** ~9 hours of hands-on work. Run them in order.

---

## What You Will Build

By the end of the series you will have built — from scratch — a complete RAG pipeline:

```
Raw text
   |
   v
[Preprocessing] -- tokenize, clean, remove stopwords
   |
   v
[BM25 index] + [SBERT FAISS index]
   |                    |
   +-----> RRF fusion <-+       <-- Hybrid retrieval
               |
               v
     [Cross-encoder re-ranker]  <-- Re-ranking
               |
               v
      [Prompt builder]          <-- Augmentation
     "[4:36] And worship..."
               |
               v
     [flan-t5-small LLM]        <-- Generation
               |
               v
    Grounded answer with citations
```

---

## Repository Structure

```
BOW-to-Embeddings/
|
|-- Claude/                  <-- Recommended series (9 notebooks, Claude Assisted)
|   |-- NB01_Text_Preprocessing_and_BoW.ipynb
|   |-- NB02_TF_IDF_and_Ranked_Retrieval.ipynb
|   |-- NB03_NGrams_and_Context.ipynb
|   |-- NB04_Word2Vec_Embeddings.ipynb
|   |-- NB05_FastText_and_Subword_Embeddings.ipynb
|   |-- NB06_Sentence_Embeddings_and_Semantic_Search.ipynb
|   |-- NB07_Vector_Databases_and_FAISS.ipynb
|   |-- NB08_RAG_and_LLM_Retrieval.ipynb
|   |-- NB09_Advanced_RAG_Techniques.ipynb
|   `-- *.png                 (generated visualizations)
|
|-- codex/                   <-- Parallel implementation (Codex/GPT-4)
|-- cursor/                  <-- Parallel implementation (Cursor AI)
|
|-- quran_translations.csv   <-- Dataset (6,236 verses, 18 translations)
|-- requirements.txt         <-- All dependencies
`-- README.md
```

The `codex/` and `cursor/` folders contain independent implementations of the same workshop built with different AI tools. Comparing all three series is a useful exercise for understanding how different tools approach the same pedagogical goals.

---

## Prerequisites

| Knowledge | Required? |
|---|---|
| Python basics (loops, functions, lists) | Yes |
| pandas and numpy familiarity | Helpful |
| Machine learning background | Not required |
| GPU | Not required (all notebooks run on CPU) |

---

## Environment Setup

The workshop uses a local virtual environment. All packages are pinned in `requirements.txt` for reproducibility.

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note for Windows users:** PyTorch CPU-only is recommended to avoid DLL errors:

```powershell
pip install torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu `
    --index-url https://download.pytorch.org/whl/cpu
```

**First-run downloads:** Some notebooks download models automatically on first use:
- NLTK punkt tokenizer (~1 MB)
- `all-MiniLM-L6-v2` sentence encoder (~90 MB)
- `google/flan-t5-small` generator (~300 MB)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (~85 MB)

---

## Dataset

The workshop uses a single CSV file with 6,236 rows:

```
quran_translations.csv
Columns: Surah, Verse, ahmedali, ahmedraza, arberry, daryabadi, hilali,
         itani, maududi, mubarakpuri, pickthall, qarai, qaribullah,
         sahih, sarwar, shakir, wahiduddin, yusufali
```

All notebooks use only the `daryabadi` column. The dataset is included in the repository — no download needed.

---

## Key Libraries

| Library | Used in | Purpose |
|---|---|---|
| `nltk` | NB01–NB09 | Tokenization, stopwords, sentence splitting |
| `scikit-learn` | NB01–NB03 | CountVectorizer, TfidfVectorizer, cosine similarity |
| `gensim` | NB04–NB05 | Word2Vec, FastText training |
| `sentence-transformers` | NB06–NB09 | SBERT encoding, CrossEncoder re-ranking |
| `faiss-cpu` | NB07–NB09 | Vector indexing and approximate nearest neighbor |
| `transformers` | NB08–NB09 | flan-t5-small generation |
| `rank_bm25` | NB09 | BM25 sparse retrieval |

---

## How to Use These Notebooks

1. **Run cells in order.** Later cells depend on variables set earlier.
2. **Read the markdown before running code.** Each section explains the *why* before the *how*.
3. **Try your own queries.** Every retrieval demo has a `query` variable — change it.
4. **Check the reflection questions.** Each notebook ends with 5 questions. Try to answer them before looking at the answers.
5. **Do not skip notebooks.** Each one introduces a concept that the next one builds on or critiques.

---

## Intended Audience

- University students taking an NLP or IR course
- Developers wanting a practical introduction to embeddings and RAG
- Instructors looking for a historically structured teaching sequence
- Anyone curious about how modern search and LLM systems actually work

---

## About the Claude Series

The notebooks in `Claude/` were developed iteratively with Claude (Anthropic), cross-evaluated against the `codex/` and `cursor/` implementations, and patched with the best ideas from all three. Each notebook title includes **"Claude Assisted"** to reflect this.
