from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "NLP_Workshop_06_Sentence_Embeddings_and_Semantic_Search.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
cells = []

cells.append(
    md(
        """# NLP Workshop 06: Sentence Embeddings and Semantic Search

Codex Assisted

This notebook continues the workshop using the same dataset and the same translation column:

```python
text_column = "daryabadi"
```

This notebook makes another major conceptual shift:

> we move from word-level representations to whole-sentence and whole-verse representations.

That shift is essential if we want to build a real semantic search system.

---

## Workshop Roadmap

```text
NB01  Text Preprocessing and Bag of Words
NB02  TF-IDF and Ranked Retrieval
NB03  N-Grams and Context
NB04  Word2Vec Embeddings
NB05  FastText and Subword Embeddings
NB06  Sentence Embeddings and Semantic Search   <- You are here
NB07  Vector Databases and FAISS
NB08  RAG and LLM-based Retrieval
```

Earlier notebooks improved lexical and word-level modeling. But users usually search with full phrases, not just single words. To retrieve meaningfully related verses, we need representations for **whole texts**.
"""
    )
)

cells.append(
    md(
        """## Learning Outcomes

By the end of this notebook, students should be able to:

- explain why word-level embeddings are not enough for full retrieval
- describe what sentence embeddings are
- understand the role of SBERT-style models
- generate sentence embeddings for Quran verses
- compute semantic similarity between a user query and all verses
- build a mini semantic search engine
- compare semantic retrieval with earlier lexical systems
- understand why sentence embeddings are central to modern search pipelines

## Prerequisite Mindset

Keep asking:

1. Why is averaging or comparing words alone not enough for full verse meaning?
2. What does it mean to embed a whole sentence in one vector?
3. Why can semantic retrieval succeed even when exact keywords do not overlap?
"""
    )
)

cells.append(
    md(
        """## Part A: Why Word-Level Embeddings Were Still Not Enough

Word2Vec and FastText improved word representations, but they still left an important problem:

- users often search with full phrases or ideas
- verses contain meaning distributed across multiple words
- semantic retrieval needs a representation for the entire text, not only individual tokens

For example, consider this query:

```text
kindness to parents
```

A useful verse might not contain those exact words.

It may use expressions like:

- kindness
- goodness
- benevolence
- being dutiful
- mother and father

This is where sentence embeddings become important.
"""
    )
)

cells.append(
    md(
        """### Why Mean Pooling of Word Vectors Is Still Weak

A common intermediate idea is:

```text
sentence embedding = average of word embeddings
```

This can be useful as a rough baseline, but it has important weaknesses:

- it treats sentence meaning as a simple average
- it weakens word order and structure
- it may let a few anchor words dominate
- it does not explicitly train for sentence similarity

This is one reason SBERT-style models became so important: they are built to produce vectors that work well for **whole-text similarity**, not just word-level composition.
"""
    )
)

cells.append(
    md(
        """## Part B: What Are Sentence Embeddings?

A **sentence embedding** is a dense vector representation for an entire sentence, paragraph, or short document.

Instead of embedding each word separately, we embed the whole text as one vector:

```text
whole sentence  ->  one dense vector
```

This gives us a representation that is much more suitable for:

- semantic search
- paraphrase detection
- clustering
- retrieval
- matching questions with relevant passages
"""
    )
)

cells.append(
    md(
        """## Part C: Why SBERT Matters

One of the most influential sentence-embedding families is **Sentence-BERT (SBERT)**.

SBERT is built to produce embeddings that work well for similarity tasks.

That means:

- similar texts should have nearby vectors
- paraphrases should be closer together
- semantically related texts should rank higher

This is a major improvement over earlier lexical systems because similarity is no longer based only on exact token overlap.
"""
    )
)

cells.append(
    md(
        """### SBERT vs Raw BERT

Raw BERT is a powerful transformer, but it was not originally designed as a ready-made sentence similarity engine.

SBERT changes the workflow by making sentence-level comparison much more practical:

- each text is encoded once into a vector
- cosine similarity can be computed efficiently
- retrieval becomes much more scalable than repeated pairwise cross-encoding

For this workshop, the key takeaway is simple:

> SBERT is designed to make semantic similarity usable in retrieval systems.
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

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer

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

We continue with the same verse dataset and the same translation column so the improvement comes from the representation, not a data change.
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
        """## Part E: Choose a Sentence Embedding Model

For educational use, we will use a compact SBERT-style model from `sentence-transformers`.

Why a compact model?

- it is easier to run in a workshop setting
- it still demonstrates the core idea
- students can see practical semantic retrieval without needing a huge infrastructure setup

If the model is available locally or downloadable in the environment, the code below will work directly.
"""
    )
)

cells.append(
    md(
        """### Runtime and Deployment Note

Sentence embedding models are much more powerful than TF-IDF, but they are also heavier.

Important practical considerations:

- encoding all verses takes more time than fitting a simple lexical vectorizer
- CPU inference is slower than GPU inference
- storing many dense vectors uses memory
- production systems often precompute embeddings once, then reuse them

This matters because modern NLP is not only about accuracy. It is also about engineering cost.
"""
    )
)

cells.append(
    code(
        """model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(model_name)

print("Sentence embedding model loaded:")
print(model_name)"""
    )
)

cells.append(
    md(
        """## Part F: Encode All Verses

Now we convert every verse into a sentence embedding.

This is the first step in building a semantic retrieval system:

```text
Verse text
   ->
Sentence embedding
   ->
Vector index or similarity search
```
"""
    )
)

cells.append(
    code(
        """verse_texts = work_df[text_column].tolist()
verse_embeddings = embedding_model.encode(
    verse_texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

print("Embedding matrix shape:", verse_embeddings.shape)"""
    )
)

cells.append(
    md(
        """## Part G: Inspect a Sentence Embedding

Like word embeddings, sentence embeddings are dense vectors.

But now each vector represents the meaning of a whole verse rather than a single token.
"""
    )
)

cells.append(
    code(
        """sample_index = 0
sample_verse = work_df.loc[sample_index, text_column]
sample_embedding = verse_embeddings[sample_index]

print("Sample verse:")
print(sample_verse)
print("\\nEmbedding shape:", sample_embedding.shape)
print("First 15 dimensions:")
print(sample_embedding[:15])"""
    )
)

cells.append(
    md(
        """## Part H: Build a Semantic Search Engine

Now we create the core retrieval pipeline:

```text
User Query
   ->
Sentence Embedding
   ->
Cosine Similarity Against Verse Embeddings
   ->
Ranked Results
```

This is the basic form of dense retrieval.
"""
    )
)

cells.append(
    code(
        """def semantic_search(query: str, top_k: int = 5):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    similarities = cosine_similarity(query_embedding, verse_embeddings).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = work_df.loc[top_indices, ["verse_id", "Surah", "Verse", text_column]].copy()
    results["score"] = similarities[top_indices]
    results["query"] = query
    return results[["verse_id", "Surah", "Verse", text_column, "score", "query"]]


query = "kindness to parents"
semantic_results = semantic_search(query, top_k=5)

print(f"Semantic search results for: {query!r}")
semantic_results"""
    )
)

cells.append(
    md(
        """## Part I: Why This Is Different from Earlier Search

In earlier notebooks, retrieval depended heavily on:

- exact terms
- term weighting
- short phrase overlap

Now the system can retrieve based more on **meaning similarity** than on exact wording.

That is the central educational transition in this notebook.
"""
    )
)

cells.append(
    md(
        """## Part J: Add a TF-IDF Baseline on the Same Verses

To understand the value of semantic retrieval properly, students should compare it to a strong lexical baseline on the **same verse set**.

This is important because otherwise "semantic search" can sound impressive without a fair comparison.
"""
    )
)

cells.append(
    code(
        """from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(work_df[text_column])


def tfidf_search(query: str, top_k: int = 5):
    query_vector = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, X_tfidf).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = work_df.loc[top_indices, ["verse_id", "Surah", "Verse", text_column]].copy()
    results["score"] = similarities[top_indices]
    results["query"] = query
    return results[["verse_id", "Surah", "Verse", text_column, "score", "query"]]


baseline_query = "kindness to parents"

print("TF-IDF baseline results:")
display(tfidf_search(baseline_query, top_k=5))

print("\\nSentence embedding semantic results:")
display(semantic_search(baseline_query, top_k=5))"""
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
    display(semantic_search(q, top_k=5))"""
    )
)

cells.append(
    md(
        """## Part K: Demonstrate Semantic Generalization

A strong semantic model may retrieve relevant verses even when the exact wording differs.

This is the core reason dense retrieval became so important in modern NLP systems.
"""
    )
)

cells.append(
    code(
        """generalization_queries = [
    "kindness to parents",
    "helping the poor",
    "belief in the unseen",
]

for q in generalization_queries:
    print("=" * 100)
    print(f"SEMANTIC QUERY: {q}")
    display(semantic_search(q, top_k=3))"""
    )
)

cells.append(
    md(
        """## Part L: Visualize Verse Embeddings with PCA

Sentence embeddings live in a high-dimensional space, so direct interpretation is difficult.

We can project a small sample of verse embeddings into 2D just to get an intuition for geometric structure.
"""
    )
)

cells.append(
    code(
        """sample_n = 120
sample_embeddings = verse_embeddings[:sample_n]
sample_labels = work_df["verse_id"].iloc[:sample_n].tolist()

pca = PCA(n_components=2, random_state=42)
sample_2d = pca.fit_transform(sample_embeddings)

pca_df = pd.DataFrame(sample_2d, columns=["PC1", "PC2"])
pca_df["verse_id"] = sample_labels

plt.figure(figsize=(10, 7))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", s=35)
plt.title("PCA Projection of a Sample of Verse Embeddings")
plt.show()"""
    )
)

cells.append(
    md(
        """## Part M: Query-to-Verse Similarity Scores

A bar chart can help students see that semantic retrieval is still a ranking problem.

The difference is that the ranking is now based on dense semantic vectors rather than sparse lexical overlap.
"""
    )
)

cells.append(
    code(
        """viz_query = "kindness to parents"
viz_results = semantic_search(viz_query, top_k=10).copy()

sns.barplot(data=viz_results, x="score", y="verse_id", palette="Greens_r")
plt.title(f"Top Semantic Search Results for Query: {viz_query}")
plt.xlabel("Cosine similarity")
plt.ylabel("Verse ID")
plt.show()"""
    )
)

cells.append(
    md(
        """## Part N: Why Sentence Embeddings Changed NLP Retrieval

Sentence embeddings changed retrieval systems because they made it practical to search by meaning at the text level.

They are useful because:

- the user query becomes one dense vector
- each verse becomes one dense vector
- similarity can be computed efficiently
- semantically related results can rank highly without exact lexical overlap

This is one of the core foundations of modern dense retrieval systems.
"""
    )
)

cells.append(
    md(
        """## Part O: What Sentence Embeddings Still Do Not Solve

Sentence embeddings are powerful, but they are not perfect.

### Limitation 1: Approximate Meaning

The embedding is only a learned approximation of meaning.

### Limitation 2: Corpus and Model Bias

The model reflects the data and training assumptions behind it.

### Limitation 3: Similarity Is Not Proof

Two texts can be close in embedding space without one fully proving, entailing, or explaining the other.

This matters in practice because:

- semantic neighbors are not guaranteed evidence
- high similarity is not the same as correctness
- retrieval should support reasoning, not replace it

### Limitation 4: No Full Explanation by Itself

A similarity score tells us that two texts are close, but not always why.

### Limitation 5: Scaling and Indexing Still Matter

If we want fast retrieval over very large corpora, we need specialized indexing systems.

That leads directly to the next notebook on **FAISS and vector databases**.
"""
    )
)

cells.append(
    md(
        """## Part P: Discussion Prompts

1. Why are sentence embeddings better suited than word embeddings for verse retrieval?
2. Why can semantic search succeed when exact keyword search fails?
3. Why is semantic retrieval still not the same as full reasoning or explanation?
4. Why do we need vector indexes once the corpus becomes large?
"""
    )
)

cells.append(
    md(
        """## Part Q: Summary and Transition to Notebook 7

In this notebook, we moved from word-level embeddings to sentence-level semantic retrieval.

### What We Learned

- word embeddings are not enough for full verse meaning
- sentence embeddings represent whole texts as dense vectors
- SBERT-style models support semantic similarity well
- a semantic Quran search engine can retrieve related verses without exact overlap
- dense retrieval is a core building block of modern NLP systems

### Historical Insight

This notebook marks another major transition:

```text
from embedding words
to embedding whole texts for semantic retrieval
```

### What Comes Next?

Notebook 7 introduces **FAISS and vector indexing**.

That notebook answers the engineering question:

> once we have dense vectors, how do we search them quickly at scale?

---

## Limitation-to-Next-Step Map

| What sentence embeddings improve | What still remains weak | What comes next |
|---|---|---|
| Whole-text semantic similarity | Fast search over very large corpora | **Notebook 7: FAISS** |
| Query-to-verse semantic matching | Efficient indexing | **Vector databases** |
| Dense retrieval quality | Explanatory generation | **Notebook 8: RAG** |
| Meaning-aware ranking | Full end-user answer systems | **Retrieval + generation pipelines** |
"""
    )
)

cells.append(
    md(
        """## Mini Self-Check

- [ ] I can explain why sentence embeddings are needed after word embeddings.
- [ ] I understand the basic idea of SBERT-style semantic similarity.
- [ ] I can build a dense semantic search function with cosine similarity.
- [ ] I understand why dense retrieval is different from lexical retrieval.
- [ ] I understand why vector indexing comes next.
"""
    )
)

cells.append(
    code(
        """# Suggested student exercises
# 1. Try semantic queries that use different wording from the verse text.
# 2. Compare queries like:
#    - "kindness to parents"
#    - "charity for the needy"
#    - "guidance for the faithful"
# 3. Identify one case where semantic search succeeds beyond exact overlap.
# 4. Identify one case where the semantic match still feels imperfect.

print("Notebook 6 is complete. Review the semantic retrieval behavior before moving to FAISS.")"""
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
