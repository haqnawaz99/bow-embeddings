from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "NLP_Workshop_08_RAG_and_LLM_Based_Retrieval.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
cells = []

cells.append(
    md(
        """# NLP Workshop 08: RAG and LLM-Based Retrieval

Codex Assisted

This notebook continues the workshop using the same dataset and the same translation column:

```python
text_column = "daryabadi"
```

This is the final notebook in the workshop series.

It answers the final question in the progression:

> once we can retrieve relevant verses semantically and efficiently, how do we use retrieval to help a language model generate grounded answers?

---

## Workshop Roadmap

```text
NB01  Text Preprocessing and Bag of Words
NB02  TF-IDF and Ranked Retrieval
NB03  N-Grams and Context
NB04  Word2Vec Embeddings
NB05  FastText and Subword Embeddings
NB06  Sentence Embeddings and Semantic Search
NB07  FAISS and Vector Retrieval
NB08  RAG and LLM-Based Retrieval   <- You are here
```

This notebook brings the whole workshop together:

- text preprocessing
- lexical search
- embeddings
- semantic retrieval
- vector indexing
- grounded answer generation
"""
    )
)

cells.append(
    md(
        """## Learning Outcomes

By the end of this notebook, students should be able to:

- explain what Retrieval-Augmented Generation (RAG) is
- describe why retrieval helps LLM systems
- understand how RAG reduces hallucination risk
- build a simple retrieval-plus-generation pipeline
- separate retrieval from generation conceptually
- explain why good retrieval quality matters in LLM applications
- understand the limits of simple educational RAG systems

## Prerequisite Mindset

Keep asking:

1. Why is an LLM alone not enough for grounded retrieval tasks?
2. What is the role of retrieved context in answer generation?
3. Why is retrieval quality still important even when a language model is very powerful?
"""
    )
)

cells.append(
    md(
        """## Part A: Why Retrieval Still Matters in the LLM Era

Large language models are powerful, but they have a major weakness:

- they can generate fluent answers
- but fluency is not the same as factual grounding

If we ask a model a question about a corpus, we often want:

- relevant evidence
- traceable context
- less hallucination
- better trustworthiness

That is where **RAG** comes in.

RAG combines:

1. retrieval
2. context selection
3. generation

Instead of asking a model to answer from memory alone, we first retrieve relevant passages and provide them as grounded context.
"""
    )
)

cells.append(
    md(
        """## Part B: What Is RAG?

**Retrieval-Augmented Generation (RAG)** is a pipeline in which a system:

1. receives a user query
2. retrieves relevant passages from a corpus
3. passes those passages to a language model
4. generates an answer grounded in the retrieved context

### Core Pipeline

```text
User Query
   ->
Embed / retrieve relevant passages
   ->
Select top-k context
   ->
Build grounded prompt
   ->
LLM generates answer
```

This is one of the most important patterns in modern NLP applications.
"""
    )
)

cells.append(
    md(
        """## Part C: Why RAG Reduces Hallucination Risk

RAG does not magically eliminate hallucinations, but it often reduces them.

Why?

Because the model is not forced to rely only on its internal parameters. It is given external context to use while answering.

That means:

- retrieval improves grounding
- retrieved passages can serve as evidence
- answers become more corpus-aware

This is especially useful when we want answers tied to a specific dataset, such as the selected Quran translation verses.
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

We continue with the same verse dataset and the same translation column to keep the workshop consistent end to end.
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
        """## Part E: Rebuild the Dense Retrieval Layer

RAG begins with retrieval.

So we recreate the semantic retrieval layer using the same sentence embedding model and FAISS-style vector search approach from the previous notebooks.
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

print("Embedding matrix shape:", verse_embeddings.shape)
print("FAISS index size:", index.ntotal)"""
    )
)

cells.append(
    md(
        """## Part F: Retrieval Function

This function retrieves the top-k verses for a user query.

It is the same retrieval logic as a dense search system, but now it will feed a generation step.
"""
    )
)

cells.append(
    code(
        """def retrieve_context(query: str, top_k: int = 5):
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
    results["query"] = query
    return results[["verse_id", "Surah", "Verse", text_column, "score", "query"]]"""
    )
)

cells.append(
    md(
        """## Part G: Retrieve Relevant Context

Before any generation happens, we inspect the retrieved passages.

This is important because **RAG quality begins with retrieval quality**.
"""
    )
)

cells.append(
    code(
        """query = "kindness to parents"
retrieved_results = retrieve_context(query, top_k=5)

print(f"Retrieved context for: {query!r}")
retrieved_results"""
    )
)

cells.append(
    md(
        """## Part H: Build a Grounded Prompt

Now we convert retrieved passages into a prompt-like structure for a language model.

Even if we do not call an external LLM API inside this notebook, students must understand the prompt design pattern.

### Prompt Template

```text
Question
Retrieved context
Instruction to answer using only the context
```

This prompt structure is one of the most important practical ideas in educational RAG systems.
"""
    )
)

cells.append(
    code(
        """def build_rag_prompt(query: str, retrieved_df: pd.DataFrame) -> str:
    context_blocks = []
    for _, row in retrieved_df.iterrows():
        context_blocks.append(
            f"[Verse {row['verse_id']}] {row[text_column]}"
        )

    context_text = "\\n".join(context_blocks)

    prompt = f\"\"\"You are answering a question using retrieved Quran translation verses.

Question:
{query}

Retrieved Context:
{context_text}

Instruction:
Answer the question using only the retrieved context. If the context is insufficient, say so clearly.
\"\"\"
    return prompt


rag_prompt = build_rag_prompt(query, retrieved_results)
print(rag_prompt[:2500])"""
    )
)

cells.append(
    md(
        """## Part I: Simple Educational Generation Step

In a full production RAG system, the prompt would be sent to a language model.

For this workshop notebook, we implement a lightweight educational stand-in:

- retrieve passages
- present them
- synthesize a simple grounded answer template

This keeps the notebook executable without depending on live external API calls.
"""
    )
)

cells.append(
    code(
        """def simple_grounded_answer(query: str, retrieved_df: pd.DataFrame) -> str:
    top_context = retrieved_df.head(3)
    bullet_lines = []

    for _, row in top_context.iterrows():
        bullet_lines.append(f"- Verse {row['verse_id']}: {row[text_column]}")

    answer = f\"\"\"Question: {query}

Grounded answer draft:
The retrieved verses suggest that the answer should be grounded in themes found in the selected translation. Relevant supporting verses include:

{chr(10).join(bullet_lines)}

Educational note:
This is a retrieval-grounded summary template. In a full RAG pipeline, an LLM would turn the retrieved context into a more natural and concise final answer while still staying grounded in these passages.
\"\"\"
    return answer


print(simple_grounded_answer(query, retrieved_results))"""
    )
)

cells.append(
    md(
        """## Part J: RAG vs No-Context Comparison

One of the most important teaching moments in a RAG notebook is to contrast:

- answering with retrieved context
- answering without retrieved context

Even if we use only a lightweight educational stand-in here, the conceptual contrast is valuable.
"""
    )
)

cells.append(
    code(
        """def no_context_answer(query: str) -> str:
    return f\"\"\"Question: {query}

No-context answer draft:
This answer is generated without retrieved supporting passages. It may sound fluent, but it is not grounded in specific verses from the selected corpus.
\"\"\"


print("Without retrieved context:")
print(no_context_answer(query))

print("\\nWith retrieved context:")
print(simple_grounded_answer(query, retrieved_results))"""
    )
)

cells.append(
    md(
        """## Part K: Compare Multiple RAG Queries

This section helps students see that RAG is not one fixed answer. It is a retrieval-and-context pattern that can be reused for many different questions.
"""
    )
)

cells.append(
    code(
        """rag_queries = [
    "kindness to parents",
    "guidance for believers",
    "mercy and forgiveness",
    "charity for the needy",
]

for q in rag_queries:
    print("=" * 100)
    print(f"RAG QUERY: {q}")
    retrieved = retrieve_context(q, top_k=3)
    display(retrieved)
    print(simple_grounded_answer(q, retrieved))
    print()"""
    )
)

cells.append(
    md(
        """## Part L: Reranking Concept

Many real RAG systems do not stop after the first retrieval stage.

A common pattern is:

```text
retrieve many candidates
   ->
rerank them with a stronger model
   ->
pass the best few to the generator
```

Why rerank?

- the first retrieval stage is usually optimized for speed
- a second stage can be slower but more precise
- this often improves the final context quality

In a production system, reranking can be done with cross-encoders or other stronger scoring models.
"""
    )
)

cells.append(
    md(
        """## Part M: Simple Grounding Check

Even a basic educational RAG system should encourage students to ask:

> does the generated answer actually rely on the retrieved passages?

We implement a small toy check by listing which verse IDs were retrieved and reminding students that grounded answers should stay tied to those sources.
"""
    )
)

cells.append(
    code(
        """def grounding_check(retrieved_df: pd.DataFrame) -> pd.DataFrame:
    check_df = retrieved_df[["verse_id", "score"]].copy()
    check_df["used_as_context"] = True
    return check_df


print("Grounding check table:")
display(grounding_check(retrieved_results))"""
    )
)

cells.append(
    md(
        """## Part N: Why Retrieval Quality Still Matters

A language model can only work well with the context it receives.

If retrieval is weak:

- the answer may miss the right evidence
- the answer may sound fluent but be weakly grounded
- the generation step may amplify retrieval mistakes

This means RAG systems still depend heavily on good retrieval foundations:

- strong embeddings
- relevant ranking
- good indexing
- careful prompt design
"""
    )
)

cells.append(
    md(
        """## Part O: RAG vs Pure LLM Use

This distinction is important.

### Pure LLM Use

```text
User asks a question
LLM answers from internal parameters
```

### RAG

```text
User asks a question
System retrieves relevant passages
LLM answers using retrieved context
```

RAG is often preferred when:

- grounding matters
- corpus-specific knowledge matters
- traceability matters
- hallucination reduction matters
"""
    )
)

cells.append(
    md(
        """## Part P: What Simple Educational RAG Still Does Not Solve

This notebook builds a minimal educational RAG pattern, but real systems are more complex.

### Limitation 1: Retrieval Errors Still Matter

If the wrong passages are retrieved, the final answer may still be poor.

### Limitation 2: Prompting Matters

How we present the retrieved context affects the final generation quality.

### Limitation 3: Context Window Limits and Truncation

If too much context is retrieved, a downstream model may truncate the prompt.

That means:

- some evidence may be cut off
- ordering of passages matters
- chunk selection becomes important

### Limitation 4: Chunking Matters

Real RAG systems often retrieve smaller chunks rather than full documents.

Chunking design affects:

- recall
- prompt length
- answer grounding

### Limitation 5: Generation Can Still Drift

Even with context, a model may overgeneralize or add unsupported details.

### Limitation 6: Hybrid Retrieval Is Often Better

In practice, many systems combine:

- dense retrieval
- lexical retrieval
- filtering
- reranking

### Limitation 7: Production Systems Need More Safety and Evaluation

Real systems need:

- evaluation
- guardrails
- citation design
- logging
- error analysis
"""
    )
)

cells.append(
    md(
        """## Part Q: Discussion Prompts

1. Why is RAG often safer than asking an LLM to answer from memory alone?
2. Why does retrieval quality still matter in a RAG system?
3. Why is a retrieved passage not automatically the same thing as a complete answer?
4. What additional components would a production RAG system need beyond this notebook?
"""
    )
)

cells.append(
    md(
        """## Part R: Final Workshop Summary

This notebook closes the historical and practical arc of the workshop.

### The Full Evolution

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
```

### What Students Should Now Be Able to Do

- preprocess text
- build Bag of Words models
- use TF-IDF
- compare lexical and semantic retrieval
- understand word and sentence embeddings
- build dense semantic search
- index vectors with FAISS
- explain and prototype a simple RAG pipeline

This is the end of the workshop sequence, but also the beginning of many advanced directions in modern NLP.
"""
    )
)

cells.append(
    md(
        """## Mini Self-Check

- [ ] I can explain what RAG is in plain language.
- [ ] I understand why retrieval helps reduce hallucination risk.
- [ ] I can describe the basic retrieval -> prompt -> generation pipeline.
- [ ] I understand why good retrieval quality is essential for good RAG.
- [ ] I can explain how this final notebook connects the whole workshop together.
"""
    )
)

cells.append(
    code(
        """# Suggested student exercises
# 1. Try new user questions and inspect the retrieved verses carefully.
# 2. Rewrite the grounded prompt template to be stricter or more concise.
# 3. Compare a strong retrieval case with a weak retrieval case.
# 4. Discuss how you would connect this notebook to a real LLM API in a production system.

print("Notebook 8 is complete. This finishes the workshop series.")"""
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
