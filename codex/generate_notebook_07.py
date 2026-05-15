from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "NLP_Workshop_07_FAISS_and_Vector_Retrieval.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
cells = []

cells.append(
    md(
        """# NLP Workshop 07: FAISS and Vector Retrieval

Codex Assisted

This notebook continues the workshop using the same dataset and the same translation column:

```python
text_column = "daryabadi"
```

This notebook answers an important engineering question:

> once we have sentence embeddings, how do we search them efficiently?

In Notebook 6, we built dense semantic retrieval by comparing a query embedding with all verse embeddings. That works for a small corpus. But as the corpus grows, we need better indexing strategies.

---

## Workshop Roadmap

```text
NB01  Text Preprocessing and Bag of Words
NB02  TF-IDF and Ranked Retrieval
NB03  N-Grams and Context
NB04  Word2Vec Embeddings
NB05  FastText and Subword Embeddings
NB06  Sentence Embeddings and Semantic Search
NB07  FAISS and Vector Retrieval   <- You are here
NB08  RAG and LLM-based Retrieval
```
"""
    )
)

cells.append(
    md(
        """## Learning Outcomes

By the end of this notebook, students should be able to:

- explain why dense retrieval needs vector indexing
- describe what FAISS is and why it matters
- build a FAISS index for verse embeddings
- run nearest-neighbor search over the index
- compare indexed retrieval with direct similarity search
- understand the difference between exact and approximate vector search
- explain why vector databases became central to modern NLP systems

## Prerequisite Mindset

Keep asking:

1. Why is semantic search alone not enough at scale?
2. What problem does an index solve?
3. Why do modern retrieval systems separate embedding generation from fast vector lookup?
"""
    )
)

cells.append(
    md(
        """## Part A: Why Notebook 6 Was Still Not Enough

In Notebook 6, we retrieved verses by:

1. embedding the query
2. comparing it against every verse embedding
3. sorting by similarity

That is perfectly fine for a small educational corpus.

But imagine:

- 100,000 documents
- 1 million documents
- 100 million documents

Brute-force comparison becomes expensive.

That is why modern dense retrieval systems rely on **vector indexes**.
"""
    )
)

cells.append(
    md(
        """## Part B: What Is FAISS?

**FAISS** stands for:

```text
Facebook AI Similarity Search
```

It is a library for efficient similarity search over dense vectors.

FAISS helps us:

- store vector embeddings
- search for nearest neighbors quickly
- scale beyond tiny corpora
- support exact or approximate search strategies

This is one of the key technologies behind modern semantic retrieval systems.

FAISS itself is a **vector search library**, not a full end-to-end document database.

In practice, production systems often combine:

- embeddings
- FAISS or another ANN engine
- metadata storage
- filtering logic
- application APIs
"""
    )
)

cells.append(
    md(
        """## Part C: Exact vs Approximate Search

There are two big ideas in vector retrieval:

### Exact Search

Search every vector and compute the true nearest neighbors.

Pros:

- accurate
- simple

Cons:

- slower at very large scale

### Approximate Search

Search intelligently so we can find very good neighbors without checking every vector exhaustively.

Pros:

- much faster at scale

Cons:

- may not always return the exact mathematically best result

For this workshop notebook, we start with a simple FAISS exact index so students understand the core mechanics first.
"""
    )
)

cells.append(
    code(
        """from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    display = print

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

print("Libraries imported successfully.")"""
    )
)

cells.append(
    md(
        """## Part D: Load the Dataset

We continue with the same verse dataset and the same translation column to keep the representation story consistent across notebooks.
"""
    )
)

cells.append(
    code(
        """DATA_PATH = Path("../quran_translations.csv")
text_column = "daryabadi"

df = pd.read_csv(DATA_PATH)
required_columns = ["Surah", "Verse", text_column]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

work_df = df[required_columns].copy()
work_df = work_df.dropna(subset=[text_column]).reset_index(drop=True)
work_df["verse_id"] = work_df["Surah"].astype(str) + ":" + work_df["Verse"].astype(str)

print("Working dataframe shape:", work_df.shape)
work_df.head()"""
    )
)

cells.append(
    md(
        """## Part E: Recreate the Verse Embeddings

To build a FAISS index, we first need the dense verse embeddings from Notebook 6.

We will use the same sentence-transformer model so the vector space stays consistent.
"""
    )
)

cells.append(
    code(
        """model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(model_name)

verse_texts = work_df[text_column].tolist()
verse_embeddings = embedding_model.encode(
    verse_texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

print("Embedding matrix shape:", verse_embeddings.shape)
print("Embedding dtype before FAISS:", verse_embeddings.dtype)"""
    )
)

cells.append(
    md(
        """## Part F: Why Normalization Matters

In Notebook 6, we used cosine similarity.

FAISS can work with different distance measures. A common trick is:

- normalize embeddings
- use inner product search

When vectors are normalized, inner product and cosine similarity become closely aligned.
"""
    )
)

cells.append(
    code(
        """verse_embeddings = verse_embeddings.astype("float32")
faiss.normalize_L2(verse_embeddings)

print("Embedding dtype after conversion:", verse_embeddings.dtype)
print("First vector norm:", np.linalg.norm(verse_embeddings[0]))"""
    )
)

cells.append(
    md(
        """## Part G: Build a FAISS Index

We start with a simple exact search index:

```python
faiss.IndexFlatIP
```

`IP` stands for inner product.

Because our vectors are normalized, this behaves like cosine-based nearest-neighbor search.
"""
    )
)

cells.append(
    code(
        """embedding_dim = verse_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(verse_embeddings)

print("FAISS index built successfully.")
print("Number of vectors in index:", index.ntotal)
print("Embedding dimension:", embedding_dim)"""
    )
)

cells.append(
    md(
        """## Part H: A Direct Brute-Force Search Baseline

Before searching with FAISS, we keep a direct baseline so students can compare the two approaches conceptually.
"""
    )
)

cells.append(
    code(
        """def brute_force_semantic_search(query: str, top_k: int = 5):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    similarities = cosine_similarity(query_embedding, verse_embeddings).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = work_df.loc[top_indices, ["verse_id", "Surah", "Verse", text_column]].copy()
    results["score"] = similarities[top_indices]
    results["method"] = "Brute Force"
    results["query"] = query
    return results[["method", "verse_id", "Surah", "Verse", text_column, "score", "query"]]"""
    )
)

cells.append(
    md(
        """## Part I: FAISS Search Function

Now we build the indexed retrieval function.

This is the core engineering move of the notebook:

```text
Query
   ->
Embed once
   ->
Search FAISS index
   ->
Return nearest vectors
```
"""
    )
)

cells.append(
    code(
        """def faiss_search(query: str, top_k: int = 5):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, top_k)

    top_indices = indices[0]
    top_scores = scores[0]

    results = work_df.loc[top_indices, ["verse_id", "Surah", "Verse", text_column]].copy()
    results["score"] = top_scores
    results["method"] = "FAISS"
    results["query"] = query
    return results[["method", "verse_id", "Surah", "Verse", text_column, "score", "query"]]"""
    )
)

cells.append(
    md(
        """## Part J: Mini Vector Search Engine

We can now search the indexed verse collection with semantic queries.
"""
    )
)

cells.append(
    code(
        """query = "kindness to parents"

print("Brute-force dense retrieval:")
display(brute_force_semantic_search(query, top_k=5))

print("\\nFAISS indexed retrieval:")
display(faiss_search(query, top_k=5))"""
    )
)

cells.append(
    md(
        """## Part K: Compare Brute Force vs FAISS Results

For this exact index, the returned results should be very similar or identical to the brute-force baseline.

That is a useful lesson:

- FAISS is not changing the meaning representation
- FAISS is changing **how we search the vector space efficiently**
"""
    )
)

cells.append(
    code(
        """comparison_queries = [
    "kindness to parents",
    "forgiveness and mercy",
    "guidance for believers",
    "day of judgment",
]

for q in comparison_queries:
    print("=" * 100)
    print(f"QUERY: {q}")
    print("-" * 100)
    display(brute_force_semantic_search(q, top_k=3))
    display(faiss_search(q, top_k=3))"""
    )
)

cells.append(
    md(
        """## Part L: Add an Approximate FAISS Index with IVFFlat

So far, we used an exact index.

Now we introduce a simple approximate index:

```python
faiss.IndexIVFFlat
```

This is important because Notebook 7 should not only show indexing. It should also show the beginning of the **speed vs accuracy tradeoff**.

### Two Important Parameters

- `nlist`: how many coarse clusters the space is partitioned into
- `nprobe`: how many of those clusters we search at query time

Intuition:

- larger `nlist` gives finer partitioning
- larger `nprobe` searches more of the space and usually improves recall
- but larger `nprobe` also costs more time
"""
    )
)

cells.append(
    code(
        """nlist = 25
quantizer = faiss.IndexFlatIP(embedding_dim)
ivf_index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)

ivf_index.train(verse_embeddings)
ivf_index.add(verse_embeddings)
ivf_index.nprobe = 5

print("IVFFlat index built successfully.")
print("Is trained:", ivf_index.is_trained)
print("Vectors in IVFFlat index:", ivf_index.ntotal)
print("nlist:", nlist)
print("nprobe:", ivf_index.nprobe)"""
    )
)

cells.append(
    code(
        """def ivf_search(query: str, top_k: int = 5):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    faiss.normalize_L2(query_embedding)
    scores, indices = ivf_index.search(query_embedding, top_k)

    top_indices = indices[0]
    top_scores = scores[0]

    results = work_df.loc[top_indices, ["verse_id", "Surah", "Verse", text_column]].copy()
    results["score"] = top_scores
    results["method"] = "FAISS-IVFFlat"
    results["query"] = query
    return results[["method", "verse_id", "Surah", "Verse", text_column, "score", "query"]]"""
    )
)

cells.append(
    md(
        """## Part M: Recall@K Against Brute Force

Approximate search should be evaluated against a stronger baseline.

One simple metric is **recall@K**:

```text
How many of the true top-K results did the approximate index recover?
```

This helps students see that approximate search is not random. It is a tradeoff between:

- speed
- memory
- retrieval quality
"""
    )
)

cells.append(
    code(
        """def recall_at_k(exact_ids, approx_ids):
    exact_set = set(exact_ids)
    approx_set = set(approx_ids)
    return len(exact_set.intersection(approx_set)) / len(exact_set)


recall_queries = [
    "kindness to parents",
    "forgiveness and mercy",
    "guidance for believers",
    "day of judgment",
]

rows = []
for q in recall_queries:
    exact = faiss_search(q, top_k=5)
    approx = ivf_search(q, top_k=5)
    rows.append(
        {
            "query": q,
            "recall_at_5": recall_at_k(exact["verse_id"].tolist(), approx["verse_id"].tolist()),
        }
    )

recall_df = pd.DataFrame(rows)
recall_df"""
    )
)

cells.append(
    code(
        """sns.barplot(data=recall_df, x="recall_at_5", y="query", palette="Oranges_r")
plt.title("IVFFlat Recall@5 Compared with Exact FAISS Search")
plt.xlabel("Recall@5")
plt.ylabel("Query")
plt.xlim(0, 1.05)
plt.show()"""
    )
)

cells.append(
    md(
        """## Part N: Simple Timing Comparison

With only a few thousand verses, both methods may feel fast.

But the timing structure still teaches an important engineering lesson:

- brute force scales linearly with corpus size
- indexes are built to make large-scale retrieval practical
"""
    )
)

cells.append(
    code(
        """import time

timing_query = "belief in the unseen"
top_k = 5

start = time.perf_counter()
_ = brute_force_semantic_search(timing_query, top_k=top_k)
brute_time = time.perf_counter() - start

start = time.perf_counter()
_ = faiss_search(timing_query, top_k=top_k)
faiss_time = time.perf_counter() - start

start = time.perf_counter()
_ = ivf_search(timing_query, top_k=top_k)
ivf_time = time.perf_counter() - start

timing_df = pd.DataFrame(
    {
        "method": ["Brute Force", "FAISS-Exact", "FAISS-IVFFlat"],
        "seconds": [brute_time, faiss_time, ivf_time],
    }
)

timing_df"""
    )
)

cells.append(
    code(
        """sns.barplot(data=timing_df, x="method", y="seconds", palette="Blues_r")
plt.title("Simple Timing Comparison on the Current Corpus")
plt.ylabel("Seconds")
plt.xlabel("Search method")
plt.show()"""
    )
)

cells.append(
    md(
        """## Part O: Why Vector Indexing Changed Retrieval Engineering

Once dense embeddings became common, search systems needed infrastructure that could:

- store huge vector collections
- retrieve nearest neighbors quickly
- support scalable semantic search

This is why FAISS and vector databases became central to modern NLP applications.

Without vector indexing, dense retrieval would remain conceptually elegant but operationally limited.
"""
    )
)

cells.append(
    md(
        """## Part P: What FAISS Still Does Not Solve

FAISS is powerful, but it solves only part of the full system.

### Limitation 1: FAISS Does Not Create Meaning

It indexes vectors. It does not learn the embeddings itself.

### Limitation 2: Retrieval Is Not Explanation

Nearest neighbors are useful, but the user may still want a grounded answer or summary.

### Limitation 3: Approximate Indexes Involve Tradeoffs

At larger scale, approximate search can trade some precision for much more speed.

### Limitation 4: FAISS Is Not a Full Metadata-Aware Database

FAISS can search vectors efficiently, but production systems often still need:

- metadata filtering
- permissions
- document storage
- hybrid ranking logic

So FAISS is an important component, not the whole application stack.

### Limitation 5: End-User Applications Need More Than Retrieval

Modern systems often combine retrieval with generation, reasoning, or question answering.

That is the bridge to Notebook 8 on **RAG and LLM-based retrieval**.
"""
    )
)

cells.append(
    md(
        """## Part Q: Discussion Prompts

1. Why does dense retrieval need a vector index when the corpus gets large?
2. Why can FAISS return the same semantic neighbors as brute-force search while still being architecturally different?
3. What is the difference between exact and approximate search?
4. Why is retrieval still not the same thing as answering a user question?
"""
    )
)

cells.append(
    md(
        """## Part R: Summary and Transition to Notebook 8

In this notebook, we moved from semantic retrieval as a concept to scalable dense retrieval as an engineering system.

### What We Learned

- sentence embeddings enable semantic search
- FAISS indexes those dense vectors for efficient nearest-neighbor lookup
- exact indexed search can match brute-force results while using a better retrieval architecture
- approximate indexes introduce controlled speed/quality tradeoffs
- vector indexing is essential for scaling semantic search

### Historical Insight

This notebook marks another major transition:

```text
from meaning-aware vectors
to scalable vector search infrastructure
```

### What Comes Next?

Notebook 8 introduces **RAG and LLM-based retrieval**.

That notebook asks the final workshop question:

> once we can retrieve relevant verses semantically and efficiently, how do we combine retrieval with language generation?

---

## Limitation-to-Next-Step Map

| What FAISS improves | What still remains weak | What comes next |
|---|---|---|
| Fast vector lookup | User-facing answer generation | **Notebook 8: RAG** |
| Scalable semantic search | Explanatory responses | **LLM-based retrieval pipelines** |
| Dense retrieval engineering | End-to-end question answering | **Retrieval + generation** |
| Practical nearest-neighbor search | Final grounded response synthesis | **RAG systems** |
"""
    )
)

cells.append(
    md(
        """## Mini Self-Check

- [ ] I can explain why vector indexes are needed after sentence embeddings.
- [ ] I understand what FAISS does and what it does not do.
- [ ] I can build a simple FAISS index and search it.
- [ ] I understand the difference between brute-force and indexed retrieval.
- [ ] I understand why RAG comes next.
"""
    )
)

cells.append(
    code(
        """# Suggested student exercises
# 1. Try several semantic queries and compare brute-force vs FAISS results.
# 2. Explain why the ranked neighbors are similar even though the retrieval architecture changed.
# 3. Read about approximate FAISS indexes such as IVF or HNSW and summarize the speed/accuracy tradeoff.
# 4. Think about how FAISS would help if the corpus had millions of verses or passages.

print("Notebook 7 is complete. Review the indexing ideas before moving to RAG.")"""
    )
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.x",
    },
}

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook written to: {OUTPUT_PATH}")
