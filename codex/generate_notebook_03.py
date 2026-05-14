from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "NLP_Workshop_03_NGrams_and_Context.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
cells = []

cells.append(
    md(
        """# NLP Workshop 03: N-Grams and Context

Codex Assisted

This notebook continues the workshop using the same dataset and the same translation column:

```python
text_column = "daryabadi"
```

We are still working inside the world of classical lexical NLP, but now we address a major weakness of unigram models:

> **word order matters, and single-word features often miss that fact.**

---

## Workshop Roadmap

```text
NB01  Text Preprocessing and Bag of Words
NB02  TF-IDF and Ranked Retrieval
NB03  N-Grams and Context   <- You are here
NB04  Word2Vec Embeddings
NB05  FastText and Subword Embeddings
NB06  Sentence Embeddings and Semantic Search
NB07  Vector Databases and FAISS
NB08  RAG and LLM-based Retrieval
```

Notebook 2 improved term weighting with TF-IDF. But TF-IDF with **unigrams only** still treats text mostly as isolated words. This notebook adds **local word sequence information**.
"""
    )
)

cells.append(
    md(
        """## Learning Outcomes

By the end of this notebook, students should be able to:

- explain what unigrams, bigrams, and trigrams are
- understand why word order and local context matter
- build unigram, bigram, and trigram vectorizers
- compare retrieval using unigram TF-IDF versus n-gram TF-IDF
- identify cases where n-grams improve lexical retrieval
- explain why n-grams still do not provide full semantics

## Prerequisite Mindset

Ask two questions throughout this notebook:

1. What meaning is lost when we break text into isolated words?
2. How much meaning is recovered when we keep short word sequences?

N-grams are an important historical step because they move classical NLP from pure word counting toward short-range structure.
"""
    )
)

cells.append(
    md(
        """## Part A: Why Notebook 2 Was Still Not Enough

In Notebook 2, TF-IDF improved weighting, but it still had a structural weakness:

- words were usually treated independently
- local order was mostly ignored
- short phrases could be broken into separate tokens

That creates problems for phrases such as:

```text
not good
very good
day of judgment
children of israel
```

If a system only looks at single words, these phrases can be represented too loosely.

### Historical Motivation

N-grams were introduced because researchers needed a way to preserve at least some local sequence information without abandoning simple vector-space models.

They do not capture deep meaning, but they do capture **short-range context**.
"""
    )
)

cells.append(
    code(
        """from pathlib import Path
import re
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
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

We continue with the same data source and the same translation column so that the only major conceptual change is the use of n-grams.
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
        """## Part C: Preprocessing Strategy for N-Grams

We still normalize text, but we must think carefully about preprocessing.

If we remove too much structure, then n-grams become less meaningful.

For this notebook, we will:

- lowercase
- remove punctuation
- tokenize
- remove stopwords for the main vector space comparison

Later in advanced NLP, people often rethink such preprocessing decisions because words like `not`, `of`, and `to` can matter inside phrases.
"""
    )
)

cells.append(
    code(
        """stop_words = set(stopwords.words("english"))


def basic_normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def preprocess_text(text: str, remove_stops: bool = True):
    tokens = word_tokenize(basic_normalize(text))
    if remove_stops:
        tokens = [token for token in tokens if token not in stop_words]
    return tokens


work_df["tokens"] = work_df[text_column].apply(preprocess_text)
work_df["clean_text"] = work_df["tokens"].apply(lambda tokens: " ".join(tokens))

work_df[[text_column, "tokens", "clean_text"]].head(3)"""
    )
)

cells.append(
    md(
        """## Part D: What Are N-Grams?

An **n-gram** is a sequence of `n` consecutive tokens.

### Common Cases

- **Unigram**: one token
- **Bigram**: two consecutive tokens
- **Trigram**: three consecutive tokens

### Example

Sentence:

```text
allah is most merciful
```

Unigrams:

```text
allah, is, most, merciful
```

Bigrams:

```text
allah is, is most, most merciful
```

Trigrams:

```text
allah is most, is most merciful
```

This is a simple idea, but it matters because short phrases often carry more meaning than isolated words.
"""
    )
)

cells.append(
    code(
        """example_tokens = ["allah", "is", "most", "merciful"]

unigrams = example_tokens
bigrams = [" ".join(example_tokens[i:i+2]) for i in range(len(example_tokens) - 1)]
trigrams = [" ".join(example_tokens[i:i+3]) for i in range(len(example_tokens) - 2)]

print("Tokens:   ", example_tokens)
print("Unigrams: ", unigrams)
print("Bigrams:  ", bigrams)
print("Trigrams: ", trigrams)"""
    )
)

cells.append(
    md(
        """## Part E: Why Word Order Matters

Consider these two phrases:

```text
not good
very good
```

A unigram model mainly sees the token `good` in both phrases. That may blur an important distinction.

Likewise:

```text
day judgment
judgment day
```

Even when two phrases share similar words, sequence can affect meaning, fluency, and retrieval usefulness.

N-grams do not solve language fully, but they can preserve patterns like:

- `day judgment`
- `children israel`
- `path straight`
- `most merciful`
"""
    )
)

cells.append(
    code(
        """phrase_docs = [
    "not good",
    "very good",
]

unigram_demo = TfidfVectorizer(ngram_range=(1, 1))
bigram_demo = TfidfVectorizer(ngram_range=(1, 2))

X_uni_demo = unigram_demo.fit_transform(phrase_docs).toarray()
X_bi_demo = bigram_demo.fit_transform(phrase_docs).toarray()

print("Unigram features:")
print(unigram_demo.get_feature_names_out())
print(pd.DataFrame(X_uni_demo, index=phrase_docs, columns=unigram_demo.get_feature_names_out()))

print("\\nUnigram + bigram features:")
print(bigram_demo.get_feature_names_out())
print(pd.DataFrame(X_bi_demo, index=phrase_docs, columns=bigram_demo.get_feature_names_out()))"""
    )
)

cells.append(
    md(
        """### The Stopword Trap

This notebook also reveals an important preprocessing danger.

In earlier classical pipelines, we often removed stopwords because words like `the`, `of`, and `is` looked uninformative.

But in phrase modeling, some of those words help define the phrase itself.

For example:

```text
day of judgment
children of israel
path of righteousness
```

If stopwords are removed too aggressively, the phrase structure changes:

```text
day judgment
children israel
path righteousness
```

Sometimes that is still usable. Sometimes it weakens the signal.

This is a very important historical lesson:

> preprocessing choices that help unigram models can sometimes hurt n-gram models.
"""
    )
)

cells.append(
    code(
        """stopword_demo_query = "day of judgment"

tokens_with_stopwords = word_tokenize(basic_normalize(stopword_demo_query))
tokens_without_stopwords = preprocess_text(stopword_demo_query, remove_stops=True)

print("Original query:            ", stopword_demo_query)
print("Tokens with stopwords:     ", tokens_with_stopwords)
print("Tokens after stopword drop:", tokens_without_stopwords)
print("Joined after preprocessing:", " ".join(tokens_without_stopwords))"""
    )
)

cells.append(
    md(
        """## Part F: Build Unigram, Bigram, and Trigram Vectorizers

We will compare three lexical representations:

1. unigram TF-IDF
2. unigram + bigram TF-IDF
3. unigram + bigram + trigram TF-IDF

This is a natural historical extension of Notebook 2:

```text
BoW counts words
TF-IDF weights words
N-grams keep short sequences of words
```
"""
    )
)

cells.append(
    code(
        """unigram_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
trigram_vectorizer = TfidfVectorizer(ngram_range=(1, 3))

X_uni = unigram_vectorizer.fit_transform(work_df["clean_text"])
X_bi = bigram_vectorizer.fit_transform(work_df["clean_text"])
X_tri = trigram_vectorizer.fit_transform(work_df["clean_text"])

print("Unigram matrix shape:          ", X_uni.shape)
print("Unigram + bigram matrix shape: ", X_bi.shape)
print("Up to trigram matrix shape:    ", X_tri.shape)"""
    )
)

cells.append(
    md(
        """## Part G: Vocabulary Growth with N-Grams

One benefit of n-grams is richer local context.

One cost is feature explosion.

As we move from unigrams to bigrams and trigrams:

- vocabulary size increases
- sparsity usually increases
- computation becomes heavier

This tradeoff is historically important because it shows why classical NLP kept searching for better representations.
"""
    )
)

cells.append(
    code(
        """vocab_sizes = pd.DataFrame(
    {
        "representation": ["unigram", "unigram+bigram", "up to trigram"],
        "vocabulary_size": [X_uni.shape[1], X_bi.shape[1], X_tri.shape[1]],
        "non_zero_entries": [X_uni.nnz, X_bi.nnz, X_tri.nnz],
    }
)

vocab_sizes"""
    )
)

cells.append(
    code(
        """sns.barplot(data=vocab_sizes, x="representation", y="vocabulary_size", palette="Blues_r")
plt.title("Vocabulary Growth as We Add N-Grams")
plt.xlabel("Representation")
plt.ylabel("Number of features")
plt.show()"""
    )
)

cells.append(
    md(
        """## Part H: Search Functions for N-Gram Retrieval

Now we build retrieval functions so we can compare the effect of local context on search quality.
"""
    )
)

cells.append(
    code(
        """def preprocess_query(query: str) -> str:
    return " ".join(preprocess_text(query, remove_stops=True))


def retrieve_with_vectorizer(query: str, vectorizer, matrix, label: str, top_k: int = 5):
    cleaned_query = preprocess_query(query)
    query_vector = vectorizer.transform([cleaned_query])
    similarities = cosine_similarity(query_vector, matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = work_df.loc[top_indices, ["verse_id", "Surah", "Verse", text_column]].copy()
    results["score"] = similarities[top_indices]
    results["model"] = label
    results["cleaned_query"] = cleaned_query
    return results[["model", "verse_id", "Surah", "Verse", text_column, "score", "cleaned_query"]]


def search_unigram(query: str, top_k: int = 5):
    return retrieve_with_vectorizer(query, unigram_vectorizer, X_uni, "Unigram TF-IDF", top_k)


def search_bigram(query: str, top_k: int = 5):
    return retrieve_with_vectorizer(query, bigram_vectorizer, X_bi, "Unigram+Bigram TF-IDF", top_k)


def search_trigram(query: str, top_k: int = 5):
    return retrieve_with_vectorizer(query, trigram_vectorizer, X_tri, "Up-to-Trigram TF-IDF", top_k)"""
    )
)

cells.append(
    md(
        """## Part I: Search Demonstration

This section is the practical heart of the notebook.

We compare retrieval results using:

- unigram TF-IDF
- unigram + bigram TF-IDF
- unigram + bigram + trigram TF-IDF

This lets students observe where short phrase information helps ranking.
"""
    )
)

cells.append(
    code(
        """query = "day of judgment"

print("Unigram retrieval:")
display(search_unigram(query, top_k=5))

print("\\nUnigram + bigram retrieval:")
display(search_bigram(query, top_k=5))

print("\\nUp-to-trigram retrieval:")
display(search_trigram(query, top_k=5))"""
    )
)

cells.append(
    md(
        """## Part J: Inspect Which N-Gram Features a Query Activates

N-gram models are still interpretable in a way that later dense embeddings usually are not.

For a phrase query, we can inspect the actual unigram and bigram features that become active.

This helps students see exactly what the vectorizer is "looking at".
"""
    )
)

cells.append(
    code(
        """inspect_query = "day of judgment"
inspect_clean = preprocess_query(inspect_query)

uni_query_vec = unigram_vectorizer.transform([inspect_clean]).toarray().flatten()
bi_query_vec = bigram_vectorizer.transform([inspect_clean]).toarray().flatten()

uni_active = pd.DataFrame(
    {
        "feature": unigram_vectorizer.get_feature_names_out(),
        "weight": uni_query_vec,
    }
)
uni_active = uni_active[uni_active["weight"] > 0].sort_values("weight", ascending=False)

bi_active = pd.DataFrame(
    {
        "feature": bigram_vectorizer.get_feature_names_out(),
        "weight": bi_query_vec,
    }
)
bi_active = bi_active[bi_active["weight"] > 0].sort_values("weight", ascending=False)

print("Original query:", inspect_query)
print("Preprocessed query:", inspect_clean)

print("\\nActive unigram features:")
display(uni_active)

print("\\nActive unigram + bigram features:")
display(bi_active)"""
    )
)

cells.append(
    code(
        """comparison_queries = [
    "day of judgment",
    "straight path",
    "children of israel",
    "most merciful",
]

for q in comparison_queries:
    print("=" * 100)
    print(f"QUERY: {q}")
    print("-" * 100)
    display(search_unigram(q, top_k=3))
    display(search_bigram(q, top_k=3))
    display(search_trigram(q, top_k=3))"""
    )
)

cells.append(
    md(
        """## Part K: Rank Drift from Unigrams to Bigrams

When we switch from a unigram model to a unigram+bigram model, the ranking of results often changes.

That movement is useful to inspect because it tells us that phrase information is affecting retrieval.

We will compare the top results for the same query under two models and see how the ranks drift.
"""
    )
)

cells.append(
    code(
        """rank_query = "day of judgment"

uni_rank = search_unigram(rank_query, top_k=10).copy()
bi_rank = search_bigram(rank_query, top_k=10).copy()

uni_rank["unigram_rank"] = range(1, len(uni_rank) + 1)
bi_rank["bigram_rank"] = range(1, len(bi_rank) + 1)

rank_drift = uni_rank[["verse_id", "unigram_rank", "score"]].rename(columns={"score": "unigram_score"})
rank_drift = rank_drift.merge(
    bi_rank[["verse_id", "bigram_rank", "score"]].rename(columns={"score": "bigram_score"}),
    on="verse_id",
    how="outer"
)

rank_drift = rank_drift.sort_values(["unigram_rank", "bigram_rank"], na_position="last")
rank_drift"""
    )
)

cells.append(
    md(
        """## Part L: Interpreting the Retrieval Differences

When n-gram retrieval improves, it is often because the model can preserve phrase-like signals.

For example:

- `day judgment` can be more informative than `day` and `judgment` separately
- `straight path` can be more focused than either word alone
- `children israel` can behave more like a concept fragment than a loose set of words

This is a meaningful improvement over unigram-only models.

But it is still a lexical improvement, not semantic understanding.
"""
    )
)

cells.append(
    md(
        """## Part M: Discussion Prompts

Use these prompts for reflection, class discussion, or lab notes:

1. Why might bigrams help more than trigrams in some retrieval tasks?
2. Why does adding trigrams often increase sparsity?
3. Can n-grams solve synonymy such as `kindness` versus `mercy`?
4. Why might aggressive stopword removal weaken certain phrase patterns?
"""
    )
)

cells.append(
    md(
        """## Part N: Limitations of N-Grams

N-grams improve local context, but they still have serious limitations.

### Limitation 1: No Deep Semantics

Even if the model captures short phrases, it still does not truly know that:

- `mercy`
- `compassion`
- `benevolence`

can be semantically related.

### Limitation 2: Sparse Feature Explosion

Adding bigrams and trigrams increases vocabulary size rapidly.

### Limitation 3: Short Context Window

N-grams capture only nearby token relationships. They do not model long-range meaning well.

### Limitation 4: Surface Form Dependence

They still depend heavily on exact wording.

These weaknesses motivate the next major historical step: **embeddings**.
"""
    )
)

cells.append(
    code(
        """failure_query = "kindness to parents"

print("Unigram retrieval:")
display(search_unigram(failure_query, top_k=5))

print("\\nUnigram + bigram retrieval:")
display(search_bigram(failure_query, top_k=5))

print("\\nUp-to-trigram retrieval:")
display(search_trigram(failure_query, top_k=5))"""
    )
)

cells.append(
    md(
        """## Part O: Visual Comparison of Top Scores

We will compare the top similarity scores for one phrase query across the three representations.
"""
    )
)

cells.append(
    code(
        """viz_query = "straight path"

viz_uni = search_unigram(viz_query, top_k=5).copy()
viz_bi = search_bigram(viz_query, top_k=5).copy()
viz_tri = search_trigram(viz_query, top_k=5).copy()

viz_uni["representation"] = "unigram"
viz_bi["representation"] = "unigram+bigram"
viz_tri["representation"] = "up-to-trigram"

viz_df = pd.concat([viz_uni, viz_bi, viz_tri], ignore_index=True)

sns.barplot(data=viz_df, x="score", y="verse_id", hue="representation")
plt.title(f"Top Retrieval Scores for Query: {viz_query}")
plt.xlabel("Cosine similarity")
plt.ylabel("Verse ID")
plt.show()"""
    )
)

cells.append(
    md(
        """## Part P: Summary and Transition to Notebook 4

In this notebook, we moved from isolated-word models toward short phrase models.

### What We Learned

- unigrams ignore most word order
- bigrams and trigrams preserve local context
- n-grams can improve lexical retrieval for phrase-like queries
- n-grams increase vocabulary size and sparsity
- n-grams still do not provide true semantic understanding

### Historical Insight

This notebook shows another important pattern in NLP history:

```text
counting words was not enough,
weighting words was not enough,
so researchers started preserving local word sequences.
```

### What Comes Next?

Notebook 4 introduces **Word2Vec**.

That is the next major conceptual leap:

> instead of treating words only as counts or short sequences, we start learning dense vector representations from context.

---

## Limitation-to-Next-Step Map

| What n-grams improve | What still remains weak | What comes next |
|---|---|---|
| Local word order | Deep semantics | **Notebook 4: Word2Vec** |
| Phrase-sensitive retrieval | Synonymy | **Embeddings** |
| More context than unigrams | High sparsity | **Dense vectors** |
| Better lexical phrase matching | Meaning still surface-based | **Semantic representations** |
"""
    )
)

cells.append(
    md(
        """## Mini Self-Check

- [ ] I can explain the difference between unigrams, bigrams, and trigrams.
- [ ] I understand why word order matters in retrieval.
- [ ] I can compare unigram TF-IDF with n-gram TF-IDF.
- [ ] I understand why n-grams improve local context but not deep semantics.
- [ ] I understand why this notebook motivates embeddings.
"""
    )
)

cells.append(
    code(
        """# Suggested student exercises
# 1. Try phrase queries such as:
#    - "most forgiving"
#    - "path of those"
#    - "children of israel"
#    - "day of resurrection"
# 2. Compare unigram, bigram, and trigram retrieval for each.
# 3. Identify one query where bigrams help clearly.
# 4. Identify one query where n-grams still fail because semantics are missing.

print("Notebook 3 is complete. Review the retrieval differences before moving to Word2Vec.")"""
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
