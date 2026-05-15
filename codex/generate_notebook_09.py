from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "NLP_Workshop_09_Advanced_RAG_Techniques.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
cells = []

cells.append(
    md(
        """# NLP Workshop 09: Advanced RAG Techniques

Codex Assisted

This notebook extends the original workshop sequence using the same dataset and the same translation column:

```python
text_column = "daryabadi"
```

Notebook 8 introduced a simple educational RAG pipeline.

This notebook asks the next practical question:

> why does naive RAG often fail, and what advanced retrieval techniques make it stronger?

---

## Extended Roadmap

```text
NB08  RAG and LLM-Based Retrieval
NB09  Advanced RAG Techniques   <- You are here
```

This notebook is an extension beyond the original 8-notebook core series. Its goal is to move from a toy RAG pattern toward the design ideas used in stronger real-world systems.
"""
    )
)

cells.append(
    md(
        """## Learning Outcomes

By the end of this notebook, students should be able to:

- explain why naive RAG can fail
- understand chunking and overlap strategies
- explain metadata-aware retrieval
- compare dense retrieval with hybrid retrieval ideas
- understand reranking as a second-stage retrieval improvement
- use multi-query retrieval as a recall improvement strategy
- think about grounding, evaluation, and failure analysis in RAG systems
- explain why advanced RAG is mostly about retrieval quality and context quality

## Prerequisite Mindset

Keep asking:

1. If a basic RAG system fails, where does it fail first?
2. Is the problem retrieval, chunking, ranking, prompting, or generation?
3. Which improvements help recall, and which improve precision?
"""
    )
)

cells.append(
    md(
        """## Part A: Why Naive RAG Often Fails

A simple RAG pipeline can already be useful, but it often breaks in practice.

Common failure modes include:

- wrong passages retrieved
- right passage not retrieved
- too much context in the prompt
- relevant evidence split across chunks
- lexical mismatch
- poor ranking inside the retrieved top-k
- weak grounding in the final answer

That means "using RAG" is not enough.

The real question becomes:

> how do we make retrieval and context assembly stronger?
"""
    )
)

cells.append(
    md(
        """## Part B: The Main Advanced RAG Levers

Most advanced RAG improvements fall into a few categories:

1. **Chunking**: how we split documents into retrievable pieces
2. **Hybrid retrieval**: combining lexical and dense retrieval
3. **Reranking**: refining top candidates with a stronger scorer
4. **Query expansion**: retrieving from multiple related formulations
5. **Metadata filtering**: restricting search by fields or constraints
6. **Grounding and evaluation**: checking whether answers really use the evidence

This notebook walks through each of these at an educational level.
"""
    )
)

cells.append(
    code(
        """from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
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
        """## Part C: Load the Dataset

We continue with the same verse dataset and same translation column so the retrieval experiments stay comparable across the full workshop.
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
        """## Part D: Build a Baseline Dense Retrieval Layer

We begin with the same dense retrieval setup used in Notebook 8. This gives us a baseline to improve.
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
).astype("float32")

faiss.normalize_L2(verse_embeddings)

embedding_dim = verse_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(verse_embeddings)

print("Dense retrieval baseline ready.")
print("FAISS index size:", index.ntotal)"""
    )
)

cells.append(
    code(
        """def dense_retrieve(query: str, top_k: int = 5):
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
    results["method"] = "Dense"
    return results[["method", "verse_id", "Surah", "Verse", text_column, "score"]]"""
    )
)

cells.append(
    md(
        """## Part E: Chunking Strategy

In large RAG systems, we usually do not retrieve whole books or full long documents.

Instead, we split text into **chunks**.

### Why Chunk?

- smaller pieces are easier to retrieve precisely
- prompt context windows are limited
- different parts of a document may answer different questions

In this workshop corpus, each verse is already a small chunk-like unit. That makes the dataset convenient for teaching.

### But the principle still matters

In larger corpora, chunk size and chunk overlap can change retrieval quality dramatically.
"""
    )
)

cells.append(
    code(
        """def make_sliding_chunks(text: str, window_words: int = 12, overlap_words: int = 4):
    words = text.split()
    chunks = []
    step = max(1, window_words - overlap_words)

    for start in range(0, len(words), step):
        chunk = words[start:start + window_words]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
        if start + window_words >= len(words):
            break
    return chunks


sample_text = work_df.loc[10, text_column]
sample_chunks = make_sliding_chunks(sample_text, window_words=8, overlap_words=3)

print("Original verse:")
print(sample_text)
print("\\nChunked version:")
for i, chunk in enumerate(sample_chunks, 1):
    print(f"{i}. {chunk}")"""
    )
)

cells.append(
    md(
        """## Part F: Metadata-Aware Retrieval

Real retrieval systems often need filtering:

- only certain documents
- only certain languages
- only certain users
- only certain sections

Our small workshop dataset has useful metadata already:

- `Surah`
- `Verse`

So we can simulate metadata-aware retrieval by filtering search results after retrieval.
"""
    )
)

cells.append(
    code(
        """def dense_retrieve_with_surah_filter(query: str, allowed_surahs=None, top_k: int = 5):
    results = dense_retrieve(query, top_k=50)

    if allowed_surahs is not None:
        results = results[results["Surah"].isin(allowed_surahs)]

    return results.head(top_k).reset_index(drop=True)


filtered_query = "guidance for believers"
filtered_results = dense_retrieve_with_surah_filter(filtered_query, allowed_surahs=[1, 2], top_k=5)

filtered_results"""
    )
)

cells.append(
    md(
        """## Part G: Hybrid Retrieval Idea

Dense retrieval is powerful, but lexical retrieval still helps.

In many real systems, retrieval is **hybrid**:

- dense retrieval captures semantic similarity
- lexical retrieval captures exact token overlap

This is especially useful when:

- a query contains names or rare terms
- exact wording matters
- semantic search alone misses an obvious literal match
"""
    )
)

cells.append(
    code(
        """tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(work_df[text_column])


def lexical_retrieve(query: str, top_k: int = 5):
    query_vector = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(query_vector, X_tfidf).flatten()
    top_indices = scores.argsort()[::-1][:top_k]

    results = work_df.loc[top_indices, ["verse_id", "Surah", "Verse", text_column]].copy()
    results["score"] = scores[top_indices]
    results["method"] = "TF-IDF"
    return results[["method", "verse_id", "Surah", "Verse", text_column, "score"]]"""
    )
)

cells.append(
    code(
        """hybrid_query = "day of judgment"

print("Lexical retrieval:")
display(lexical_retrieve(hybrid_query, top_k=5))

print("\\nDense retrieval:")
display(dense_retrieve(hybrid_query, top_k=5))"""
    )
)

cells.append(
    md(
        """## Part H: Simple Hybrid Merge

One easy educational hybrid strategy is:

1. retrieve lexical top-k
2. retrieve dense top-k
3. merge and deduplicate

This is not the most advanced fusion method, but it helps students understand the idea.
"""
    )
)

cells.append(
    code(
        """def simple_hybrid_retrieve(query: str, top_k_each: int = 5):
    lexical = lexical_retrieve(query, top_k=top_k_each)
    dense = dense_retrieve(query, top_k=top_k_each)

    combined = pd.concat([lexical, dense], ignore_index=True)
    combined = combined.sort_values("score", ascending=False)
    combined = combined.drop_duplicates(subset=["verse_id"])
    return combined.reset_index(drop=True)


simple_hybrid_retrieve("mercy and forgiveness", top_k_each=5).head(8)"""
    )
)

cells.append(
    md(
        """## Part I: Reranking Concept

Many strong RAG systems use a two-stage ranking process:

1. retrieve candidates quickly
2. rerank candidates more carefully

Why?

- retrieval is optimized for recall and speed
- reranking is optimized for precision

For a teaching notebook, we can simulate reranking by rescoring a small candidate pool with a second criterion.
"""
    )
)

cells.append(
    code(
        """def toy_rerank(query: str, candidate_df: pd.DataFrame):
    # Educational reranking signal:
    # reward shorter passages slightly and reward lexical overlap with the query.
    query_terms = set(re.findall(r"\\w+", query.lower()))

    reranked = candidate_df.copy()
    reranked["lexical_overlap"] = reranked[text_column].str.lower().apply(
        lambda text: len(query_terms.intersection(set(re.findall(r"\\w+", text))))
    )
    reranked["text_length"] = reranked[text_column].str.split().str.len()
    reranked["rerank_score"] = reranked["score"] + 0.02 * reranked["lexical_overlap"] - 0.0005 * reranked["text_length"]
    reranked = reranked.sort_values("rerank_score", ascending=False)
    return reranked


candidate_pool = simple_hybrid_retrieve("kindness to parents", top_k_each=5)
toy_rerank("kindness to parents", candidate_pool).head(8)"""
    )
)

cells.append(
    md(
        """## Part J: Multi-Query Retrieval

Another advanced technique is to reformulate the query in multiple ways.

Why?

Because one wording may miss passages that another wording retrieves.

For example:

```text
kindness to parents
goodness to mother and father
dutifulness to parents
```

These formulations are not identical, but they may retrieve overlapping and complementary evidence.
"""
    )
)

cells.append(
    code(
        """def multi_query_retrieve(queries, top_k_each: int = 3):
    frames = []
    for q in queries:
        df_q = dense_retrieve(q, top_k=top_k_each).copy()
        df_q["subquery"] = q
        frames.append(df_q)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("score", ascending=False)
    combined = combined.drop_duplicates(subset=["verse_id"])
    return combined.reset_index(drop=True)


multi_queries = [
    "kindness to parents",
    "goodness to mother and father",
    "dutifulness to parents",
]

multi_query_retrieve(multi_queries, top_k_each=3).head(10)"""
    )
)

cells.append(
    md(
        """## Part K: Grounding and Citation Thinking

Advanced RAG is not only about retrieving text. It is also about making answers traceable.

Students should develop the habit of asking:

- which passage supports this answer?
- is the support explicit or inferred?
- are citations shown clearly?

Even a simple system should keep source identifiers visible.
"""
    )
)

cells.append(
    code(
        """grounding_query = "mercy and forgiveness"
grounding_results = simple_hybrid_retrieve(grounding_query, top_k_each=4).head(5)

grounding_results[["verse_id", "method", "score", text_column]]"""
    )
)

cells.append(
    md(
        """## Part L: Evaluation Mindset

A production-quality RAG system needs evaluation.

Important questions include:

- did retrieval find the right evidence?
- did reranking improve the final context?
- did the answer stay grounded?
- did the system miss relevant passages?

Even if we do not build a full benchmark here, students should understand that advanced RAG requires systematic evaluation.
"""
    )
)

cells.append(
    code(
        """eval_queries = [
    "kindness to parents",
    "day of judgment",
    "charity for the needy",
]

eval_rows = []
for q in eval_queries:
    lexical_top = lexical_retrieve(q, top_k=3)["verse_id"].tolist()
    dense_top = dense_retrieve(q, top_k=3)["verse_id"].tolist()
    hybrid_top = simple_hybrid_retrieve(q, top_k_each=3)["verse_id"].head(3).tolist()

    eval_rows.append(
        {
            "query": q,
            "lexical_top_ids": lexical_top,
            "dense_top_ids": dense_top,
            "hybrid_top_ids": hybrid_top,
        }
    )

pd.DataFrame(eval_rows)"""
    )
)

cells.append(
    md(
        """## Part M: Why Advanced RAG Is Mostly Retrieval Engineering

A common beginner misconception is:

```text
If the LLM is strong enough, retrieval quality does not matter much.
```

That is usually false.

In practice, many RAG gains come from:

- better chunking
- better candidate generation
- better reranking
- better filtering
- better prompt context assembly

This is why advanced RAG is often more about **retrieval engineering** than about changing the generator itself.
"""
    )
)

cells.append(
    md(
        """## Part N: What This Notebook Still Leaves Open

This extension notebook introduces several important advanced ideas, but many real-world topics remain open:

- learned rerankers
- cross-encoders
- chunk compression
- multi-hop retrieval
- structured metadata filtering
- hybrid fusion algorithms
- offline eval sets
- human review pipelines

That is normal. Advanced RAG is a large field, and this notebook is meant to give students a principled entry point.
"""
    )
)

cells.append(
    md(
        """## Part O: Discussion Prompts

1. Why can chunking choices affect RAG quality so strongly?
2. When would hybrid retrieval outperform dense-only retrieval?
3. Why is reranking often worth the extra cost?
4. Why is evaluation essential in advanced RAG systems?
"""
    )
)

cells.append(
    md(
        """## Part P: Extended Workshop Summary

This notebook extends the original workshop by moving from basic RAG to stronger retrieval design thinking.

### What Students Should Now Understand

- naive RAG often fails for retrieval reasons
- chunking, reranking, and hybrid retrieval matter
- metadata and grounding matter
- evaluation matters
- advanced RAG is a retrieval-and-context quality problem, not just a generation problem

### Final Historical Arc

```text
Keyword Search
   ->
Bag of Words
   ->
TF-IDF
   ->
N-Grams
   ->
Word2Vec
   ->
FastText
   ->
Sentence Embeddings
   ->
FAISS
   ->
RAG
   ->
Advanced RAG
```

This gives students a clear view of how NLP evolved from simple lexical matching to modern retrieval-aware language systems.
"""
    )
)

cells.append(
    md(
        """## Mini Self-Check

- [ ] I can explain why naive RAG often fails.
- [ ] I understand chunking, hybrid retrieval, and reranking at a conceptual level.
- [ ] I know why metadata and grounding matter.
- [ ] I understand that retrieval quality is central to advanced RAG.
- [ ] I can describe at least one way to evaluate a RAG system.
"""
    )
)

cells.append(
    code(
        """# Suggested student exercises
# 1. Try new query reformulations and compare single-query vs multi-query retrieval.
# 2. Change top_k_each in hybrid retrieval and observe how the merged pool changes.
# 3. Modify the toy reranking rule and see how rankings shift.
# 4. Write a short paragraph explaining when dense-only retrieval is not enough.

print("Notebook 9 is complete. This extends the workshop into advanced RAG design.")"""
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
