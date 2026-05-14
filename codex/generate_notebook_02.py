from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "NLP_Workshop_02_TFIDF_and_Ranked_Retrieval.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
cells = []

cells.append(
    md(
        """# NLP Workshop 02: TF-IDF and Ranked Retrieval

This notebook continues the workshop using the same dataset and the same translation column:

```python
text_column = "daryabadi"
```

That consistency matters. If we change both the model and the text source at the same time, students cannot tell what caused the improvement.

---

## Why This Notebook Exists

In Notebook 1, we built:

- exact keyword search
- Bag of Words vectors
- cosine-similarity retrieval

Those methods were historically important, but they had a major weakness:

> **All words were treated too uniformly.**

If a word appears in almost every verse, it may contribute a lot to the raw count, but not necessarily a lot to the meaning of the verse.

This notebook introduces **TF-IDF**, one of the most important classical weighting schemes in NLP and information retrieval.

---

## Workshop Roadmap

```text
NB01  Text Preprocessing and Bag of Words
NB02  TF-IDF and Ranked Retrieval   <- You are here
NB03  N-Grams and Context
NB04  Word2Vec Embeddings
NB05  FastText and Subword Embeddings
NB06  Sentence Embeddings and Semantic Search
NB07  Vector Databases and FAISS
NB08  RAG and LLM-based Retrieval
```
"""
    )
)

cells.append(
    md(
        """## Learning Outcomes

By the end of this notebook, students should be able to:

- explain why Bag of Words can overweight common terms
- define **term frequency (TF)** and **inverse document frequency (IDF)**
- explain the intuition behind the TF-IDF formula
- build TF-IDF vectors with `TfidfVectorizer`
- create a ranked retrieval system using cosine similarity
- compare BoW search and TF-IDF search on the same queries
- explain what TF-IDF fixes and what it still cannot fix

## Prerequisite Mindset

Approach this notebook with two questions:

1. What problem in Bag of Words is TF-IDF trying to solve?
2. What important problems still remain unsolved even after TF-IDF?

TF-IDF is a better lexical retrieval method, not a semantic understanding system.
"""
    )
)

cells.append(
    md(
        """## Part A: Historical Motivation

Bag of Words was a major step forward because it turned text into vectors. But it still had an obvious weakness:

- common words could dominate document counts
- frequent religious vocabulary could appear in many verses
- those words were not always the most discriminative terms

Information retrieval researchers needed a way to say:

```text
This word appears in this verse often,
but it also appears in many other verses,
so maybe it is not very informative.
```

That idea led to **term weighting**.

TF-IDF became one of the most successful classical solutions because it rewards terms that are:

- important within a specific document
- but not too common across the whole corpus

### What Remained Unsolved After Notebook 1

Notebook 1 gave us a usable retrieval system, but several problems remained:

- all matched words contributed too uniformly
- common terms could dominate similarity
- longer or more repetitive verses could accumulate counts without becoming more informative

Notebook 2 addresses the **weighting problem**, not the full meaning problem.
"""
    )
)

cells.append(
    code(
        """from pathlib import Path
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    display = print

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

print("Libraries imported successfully.")"""
    )
)

cells.append(
    md(
        """## Part B: Load the Dataset

We continue with the same CSV dataset and the same translation column used in Notebook 1.

That lets us compare:

- the **same verses**
- the **same preprocessing**
- different **vectorization strategies**
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
        """## Part C: Reuse the Same Preprocessing Pipeline

To compare BoW and TF-IDF fairly, we should not change everything at once.

So we reuse the same core preprocessing logic:

- lowercase
- remove punctuation
- tokenize
- remove stopwords
- optional stemming

This means the main change in this notebook is not preprocessing. The main change is **term weighting**.
"""
    )
)

cells.append(
    code(
        """stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def basic_normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def tokenize_text(text: str):
    return word_tokenize(text)


def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]


def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]


def preprocess_text(text: str, remove_stops: bool = True, apply_stemming: bool = False):
    text = basic_normalize(text)
    tokens = tokenize_text(text)
    if remove_stops:
        tokens = remove_stopwords(tokens)
    if apply_stemming:
        tokens = stem_tokens(tokens)
    return tokens


work_df["tokens"] = work_df[text_column].apply(preprocess_text)
work_df["clean_text"] = work_df["tokens"].apply(lambda tokens: " ".join(tokens))

work_df[[text_column, "tokens", "clean_text"]].head(3)"""
    )
)

cells.append(
    md(
        """## Part D: Why Raw Counts Are Not Enough

Suppose two words appear in a verse:

- one is rare across the corpus
- the other appears in thousands of verses

Should those words contribute equally to retrieval? Usually not.

### Intuition

If a term appears in almost every document, it is often less useful for distinguishing one document from another.

That is exactly the intuition behind **inverse document frequency**.

TF-IDF combines two ideas:

- **TF**: how important is the word inside this document?
- **IDF**: how rare or informative is the word across documents?
"""
    )
)

cells.append(
    md(
        """## Part E: Tiny Worked Example Before the Full Corpus

Before using the real dataset, it helps to see TF-IDF on a tiny toy corpus.

Consider three short documents:

```text
Doc 1: mercy mercy kindness
Doc 2: mercy justice
Doc 3: justice truth
```

Now think about the word `mercy`:

- it is important in Doc 1 because it appears twice
- but it is not unique to Doc 1 because it also appears in Doc 2

Now think about the word `kindness`:

- it appears in Doc 1
- it does not appear in Doc 2 or Doc 3
- so it is more distinctive

This is exactly the behavior TF-IDF is designed to capture.
"""
    )
)

cells.append(
    code(
        """toy_docs = [
    "mercy mercy kindness",
    "mercy justice",
    "justice truth",
]

toy_vectorizer = TfidfVectorizer()
toy_matrix = toy_vectorizer.fit_transform(toy_docs)
toy_terms = toy_vectorizer.get_feature_names_out()

toy_df = pd.DataFrame(toy_matrix.toarray(), columns=toy_terms, index=["Doc1", "Doc2", "Doc3"])
toy_df"""
    )
)

cells.append(
    md(
        """## Part F: Term Frequency (TF)

**Term Frequency (TF)** measures how often a word appears in a document.

In the simplest form:

```text
TF(term, document) = number of times the term appears in the document
```

Some systems use raw counts. Others use normalized counts. The exact formula can vary.

The main idea is simple:

> If a word appears several times in a verse, it may be important for that verse.
"""
    )
)

cells.append(
    code(
        """sample_doc = work_df.loc[0, "clean_text"]
sample_tokens = sample_doc.split()
sample_counts = Counter(sample_tokens)

print("Sample preprocessed verse:")
print(sample_doc)

print("\\nTerm frequencies in this verse:")
for word, count in sample_counts.items():
    print(f"{word:15s} -> {count}")"""
    )
)

cells.append(
    md(
        """## Part G: Inverse Document Frequency (IDF)

**Document Frequency (DF)** counts in how many documents a term appears.

If a term appears in many documents, it is less distinctive.

That leads to **Inverse Document Frequency (IDF)**:

```text
IDF(term) = log( total_documents / documents_containing_term )
```

Different libraries use slightly different smoothed variants, but the intuition is the same:

- common terms receive lower weight
- rarer terms receive higher weight

### Conceptual Meaning

TF answers:

```text
How important is this word inside this verse?
```

IDF answers:

```text
How special is this word across the whole corpus?
```
"""
    )
)

cells.append(
    code(
        """document_count = len(work_df)

doc_frequency = Counter()
for tokens in work_df["tokens"]:
    for token in set(tokens):
        doc_frequency[token] += 1

sample_terms = ["allah", "mercy", "merciful", "punishment", "guidance"]

idf_demo_rows = []
for term in sample_terms:
    df_term = doc_frequency.get(term, 0)
    smoothed_idf = np.log((1 + document_count) / (1 + df_term)) + 1
    idf_demo_rows.append(
        {
            "term": term,
            "document_frequency": df_term,
            "smoothed_idf": round(smoothed_idf, 4),
        }
    )

idf_demo_df = pd.DataFrame(idf_demo_rows)
idf_demo_df"""
    )
)

cells.append(
    md(
        """## Part H: The TF-IDF Formula

In its simplest conceptual form:

```text
TF-IDF(term, document) = TF(term, document) × IDF(term)
```

This means a word gets a strong weight when:

- it matters inside the document
- but it is not overly common across the corpus

This is a major improvement over raw counts because it helps distinguish informative terms from merely frequent ones.

### What TF-IDF Improves

- reduces the dominance of very common words
- improves ranked retrieval quality
- produces more discriminative document vectors

### What TF-IDF Does Not Solve

- synonymy
- deep semantics
- long-range context
- world knowledge
"""
    )
)

cells.append(
    md(
        """## Part I: sklearn Smoothing and `idf_`

In classroom explanations, we often write:

```text
IDF(term) = log(N / df(term))
```

But `scikit-learn` usually uses a **smoothed** form:

```text
IDF(term) = log((1 + N) / (1 + df(term))) + 1
```

Why smooth it?

- to avoid division-by-zero style edge cases
- to keep values numerically stable
- to make the implementation safer in general use

For students, the key idea is:

> the exact formula may vary slightly, but the intuition remains the same: rarer terms get more weight.

You can inspect the learned IDF values through the vectorizer's `idf_` attribute.
"""
    )
)

cells.append(
    md(
        """## Part J: Build BoW and TF-IDF Representations

We will build both vector spaces so we can compare them directly on the same queries.

That comparison matters because students should not just hear that TF-IDF is better. They should **observe** where it improves ranking.
"""
    )
)

cells.append(
    code(
        """bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(work_df["clean_text"])

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(work_df["clean_text"])

print("BoW matrix shape:   ", X_bow.shape)
print("TF-IDF matrix shape:", X_tfidf.shape)

print("\\nBoW non-zero entries:", X_bow.nnz)
print("TF-IDF non-zero entries:", X_tfidf.nnz)

print("\\nSame number of documents:", X_bow.shape[0] == X_tfidf.shape[0])
print("Same vocabulary size?     ", X_bow.shape[1] == X_tfidf.shape[1])"""
    )
)

cells.append(
    code(
        """idf_values = pd.DataFrame(
    {
        "term": tfidf_vectorizer.get_feature_names_out(),
        "idf": tfidf_vectorizer.idf_,
    }
).sort_values("idf", ascending=False)

print("Terms with high IDF tend to be rarer across the corpus.")
idf_values.head(10)"""
    )
)

cells.append(
    code(
        """tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
sample_tfidf = pd.DataFrame(
    X_tfidf[:5, :12].toarray(),
    columns=tfidf_feature_names[:12],
    index=[f"Verse_{i}" for i in range(5)]
)

sample_tfidf"""
    )
)

cells.append(
    md(
        """## Part K: Inspect Important TF-IDF Weights

To build intuition, let us inspect the highest-weighted terms for a specific verse.

This is a useful teaching moment because students can see that TF-IDF is not just counting words. It is emphasizing words that are more informative for that verse.
"""
    )
)

cells.append(
    code(
        """verse_index = 0
verse_text = work_df.loc[verse_index, text_column]
verse_vector = X_tfidf[verse_index].toarray().flatten()

top_term_indices = verse_vector.argsort()[::-1][:10]
top_term_weights = pd.DataFrame(
    {
        "term": tfidf_feature_names[top_term_indices],
        "tfidf_weight": verse_vector[top_term_indices],
    }
)

print("Original verse:")
print(verse_text)

print("\\nTop TF-IDF weighted terms for this verse:")
top_term_weights"""
    )
)

cells.append(
    md(
        """## Part L: Build a TF-IDF Search Engine

Now we can build a ranked retrieval system:

```text
User Query
   ->
Preprocess Query
   ->
Convert to TF-IDF Vector
   ->
Compute Cosine Similarity
   ->
Rank All Verses
   ->
Return Top-k Results
```

This is still a classical lexical system, but it is a more informed one than raw BoW search.
"""
    )
)

cells.append(
    code(
        """def preprocess_query(query: str) -> str:
    tokens = preprocess_text(query, remove_stops=True, apply_stemming=False)
    return " ".join(tokens)


def search_bow(query: str, top_k: int = 5):
    cleaned_query = preprocess_query(query)
    query_vector = bow_vectorizer.transform([cleaned_query])
    similarities = cosine_similarity(query_vector, X_bow).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = work_df.loc[top_indices, ["verse_id", "Surah", "Verse", text_column]].copy()
    results["score"] = similarities[top_indices]
    results["method"] = "BoW"
    results["cleaned_query"] = cleaned_query
    return results[["method", "verse_id", "Surah", "Verse", text_column, "score", "cleaned_query"]]


def search_tfidf(query: str, top_k: int = 5):
    cleaned_query = preprocess_query(query)
    query_vector = tfidf_vectorizer.transform([cleaned_query])
    similarities = cosine_similarity(query_vector, X_tfidf).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = work_df.loc[top_indices, ["verse_id", "Surah", "Verse", text_column]].copy()
    results["score"] = similarities[top_indices]
    results["method"] = "TF-IDF"
    results["cleaned_query"] = cleaned_query
    return results[["method", "verse_id", "Surah", "Verse", text_column, "score", "cleaned_query"]]


query = "allah is merciful"
tfidf_results = search_tfidf(query, top_k=5)

print(f"TF-IDF search results for: {query!r}")
tfidf_results"""
    )
)

cells.append(
    md(
        """## Part M: Compare BoW and TF-IDF Retrieval

This is the core educational section.

We do not want students to simply memorize:

```text
TF-IDF is better than BoW.
```

That statement is too vague.

What students should see is:

- BoW relies on raw word overlap
- TF-IDF reduces the influence of overly common terms
- ranking can improve because informative terms get more weight
"""
    )
)

cells.append(
    code(
        """comparison_query = "allah is merciful"

print("BoW results:")
display(search_bow(comparison_query, top_k=5))

print("\\nTF-IDF results:")
display(search_tfidf(comparison_query, top_k=5))"""
    )
)

cells.append(
    code(
        """comparison_queries = [
    "mercy and forgiveness",
    "day of judgment",
    "guidance for believers",
]

for q in comparison_queries:
    print("=" * 100)
    print(f"QUERY: {q}")
    print("-" * 100)
    print("Top 3 BoW results")
    display(search_bow(q, top_k=3))
    print("Top 3 TF-IDF results")
    display(search_tfidf(q, top_k=3))"""
    )
)

cells.append(
    md(
        """## Part N: Discussion Prompts

These prompts are useful in a classroom, lab, or self-study setting:

1. If two verses share the same query words, why might TF-IDF rank them differently?
2. Why is a very common word often less useful for retrieval than a rarer one?
3. Can TF-IDF solve the difference between `mercy` and `compassion` if the exact words do not overlap?
4. Why is TF-IDF still considered a lexical model rather than a semantic model?

Students should try to answer these in plain language before moving on.
"""
    )
)

cells.append(
    md(
        """## Part O: Why TF-IDF Often Improves Ranking

If a query contains both common and informative words, TF-IDF usually helps by reducing the contribution of the common terms.

For example:

- common terms may appear in many verses
- discriminative terms may appear in fewer verses
- TF-IDF pushes the ranking to pay more attention to those discriminative terms

This often produces cleaner top-k retrieval results than BoW.

But notice the word **often**.

TF-IDF is not magic. It improves weighting, not understanding.
"""
    )
)

cells.append(
    md(
        """## Part P: Failure Cases and Remaining Limitations

TF-IDF solves an important problem, but it does not solve all major NLP problems.

### Limitation 1: Still Mostly Lexical

If the query uses:

```text
kindness
```

and the verse uses:

```text
mercy
compassion
benevolence
```

TF-IDF may still miss the semantic relationship.

### Limitation 2: Limited Context

TF-IDF does not truly understand syntax, negation, or nuanced meaning.

### Limitation 3: No Deep Semantics

Words that mean similar things are still treated as different dimensions unless they overlap lexically.

This is why the workshop will later move toward:

- n-grams for local order
- word embeddings for semantic similarity
- sentence embeddings for semantic retrieval
"""
    )
)

cells.append(
    code(
        """failure_query = "kindness"

print("BoW results for a semantically difficult query:")
display(search_bow(failure_query, top_k=5))

print("\\nTF-IDF results for the same query:")
display(search_tfidf(failure_query, top_k=5))"""
    )
)

cells.append(
    md(
        """## Part Q: Visual Comparison of Query Scores

A chart can help students see ranked retrieval more concretely.

We will compare the top TF-IDF similarity scores for one query.
"""
    )
)

cells.append(
    code(
        """viz_query = "mercy and forgiveness"
viz_results = search_tfidf(viz_query, top_k=10).copy()

sns.barplot(data=viz_results, x="score", y="verse_id", palette="Greens_r")
plt.title(f"Top TF-IDF Results for Query: {viz_query}")
plt.xlabel("Cosine similarity")
plt.ylabel("Verse ID")
plt.show()"""
    )
)

cells.append(
    md(
        """## Part R: Summary and Transition to Notebook 3

In this notebook, students moved from **raw counts** to **weighted lexical retrieval**.

### What We Learned

- BoW counts words but does not weight them intelligently
- TF-IDF downweights very common terms
- TF-IDF often improves ranked retrieval quality
- TF-IDF is still not semantic understanding

### Main Historical Insight

The progression from BoW to TF-IDF shows an important pattern in NLP history:

```text
first we count words,
then we weight words,
then we realize that weighting is still not enough.
```

### What Comes Next?

Notebook 3 introduces **n-grams**.

That notebook addresses another major weakness of Bag of Words and TF-IDF:

> word order and local context matter.

---

## Limitation-to-Next-Step Map

| What we improved in Notebook 2 | What still remains weak | Next notebook or stage |
|---|---|---|
| Better term weighting | Word order still weak | **Notebook 3: N-Grams** |
| Better ranked retrieval | Synonyms still weak | **Notebooks 4-6: Embeddings** |
| Less dominance from common words | Deep semantics still absent | **Sentence embeddings** |
| Stronger lexical search | Retrieval still based on word overlap | **Semantic search systems** |
"""
    )
)

cells.append(
    md(
        """## Mini Self-Check

- [ ] I can explain the difference between TF and IDF.
- [ ] I understand why common terms usually receive lower IDF values.
- [ ] I can build a TF-IDF search engine with `TfidfVectorizer` and cosine similarity.
- [ ] I can describe at least one case where TF-IDF improves on BoW.
- [ ] I understand why TF-IDF still does not give semantic understanding.

If those points are clear, the next historical step is to add **local context** through n-grams.
"""
    )
)

cells.append(
    code(
        """# Suggested student exercises
# 1. Try TF-IDF search with queries like:
#    - "charity and the poor"
#    - "patience in hardship"
#    - "reward and punishment"
# 2. Compare top-5 BoW and top-5 TF-IDF results for each query.
# 3. Identify one case where TF-IDF clearly improves ranking.
# 4. Identify one case where TF-IDF still fails because semantics are missing.

print("Notebook 2 is complete. Review the comparisons carefully before moving to n-grams.")"""
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
