from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "NLP_Workshop_01_Text_Preprocessing_and_BoW.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
cells = []

cells.append(
    md(
        """# NLP Workshop 01: Text Preprocessing and Bag of Words

This notebook is the first lab in a complete NLP workshop series built on **one English Quran translation column only**.

For consistency throughout the workshop, we will use:

```python
text_column = "daryabadi"
```

That choice matters. In NLP, even small preprocessing and modeling decisions change the behavior of a system. If we keep switching translation columns, students can no longer tell whether a difference comes from the **model** or from the **data**.

---

## Learning Goals

By the end of this notebook, students should be able to:

- explain what NLP is and why preprocessing exists
- load and inspect a verse-level text dataset
- clean text using classical preprocessing steps
- build a vocabulary from a corpus
- represent verses with **Bag of Words (BoW)** vectors
- understand sparsity in document-term matrices
- run **exact keyword search**
- run **Bag of Words similarity search** using cosine similarity
- identify the main limitations of keyword and BoW systems

---

## Historical Position of This Notebook

This workshop is intentionally historical. We are not starting with transformers, embeddings, or LLMs. We are starting with older methods so that students understand **why modern methods became necessary**.

The roadmap for the full series is:

```text
Keyword Search
   ->
Bag of Words
   ->
TF-IDF
   ->
N-Grams
   ->
Word Embeddings
   ->
Sentence Embeddings
   ->
Vector Search
   ->
RAG and LLM Retrieval
```

Modern NLP becomes much easier to understand when students first see what older systems can and cannot do.

---

## Workshop Roadmap

```text
NB01  Text Preprocessing and Bag of Words
NB02  TF-IDF and Ranked Retrieval
NB03  N-Grams and Context
NB04  Word2Vec Embeddings
NB05  FastText and Subword Embeddings
NB06  Sentence Embeddings and Semantic Search
NB07  Vector Databases and FAISS
NB08  RAG and LLM-based Retrieval
```

This notebook stops at **BoW plus cosine retrieval**. That is intentional. Students first need to see where lexical systems work, and where they break.
"""
    )
)

cells.append(
    md(
        """## Learning Outcomes

By the end of this notebook, students should be able to:

- explain the historical role of preprocessing and Bag of Words in NLP
- load a verse-level dataset and inspect its schema
- apply a transparent preprocessing pipeline
- build and interpret a vocabulary
- create a document-term matrix with `CountVectorizer`
- perform exact keyword search
- perform BoW-based cosine similarity retrieval
- explain why BoW fails on semantics and context

## Prerequisite Mindset

This notebook is beginner-friendly, but students will benefit most if they approach it with the following mindset:

- do not treat preprocessing as a ritual; ask what each step changes
- do not treat better retrieval scores as true understanding
- do not assume old methods are useless; they are historically foundational
- keep asking: what problem does the next method solve that this method cannot?

## Part A: What Is NLP?

**Natural Language Processing (NLP)** is the field of teaching computers to work with human language such as English, Arabic, Urdu, or any other natural language.

Examples of NLP tasks include:

- search
- text classification
- sentiment analysis
- question answering
- machine translation
- information retrieval
- summarization

In this workshop, we focus on **search and retrieval** using Quran translation verses as the main learning dataset.

---

### Why Is Human Language Hard for Computers?

Language is difficult because meaning is not carried by words alone. It depends on:

- ambiguity
- context
- word order
- negation
- style variation
- morphology

Consider these sentences:

```text
I saw the man with a telescope.
The chicken is ready to eat.
Visiting relatives can be boring.
```

Each can be interpreted in more than one way. A human reader usually resolves the ambiguity using world knowledge and context. Classical NLP systems generally cannot do that well.

### Short Historical Framing

It helps to place this notebook in a larger timeline:

- early NLP used hand-written rules and linguistic heuristics
- later systems shifted toward statistics and counting words
- Bag of Words became a practical way to turn documents into vectors
- later researchers realized that word counts alone could not capture meaning
- embeddings and semantic retrieval emerged to address those failures

This notebook lives in the **counting words** phase of NLP history.

## Part B: Why Preprocessing Matters

Computers do not naturally understand text the way humans do. To a computer, text is just a sequence of characters.

For example:

```text
"Mercy"
"mercy"
"mercy,"
```

A human sees these as closely related. A naive computer system may treat them as different strings.

That is why classical NLP pipelines usually begin with preprocessing:

```text
Raw Text
   ->
Cleaning
   ->
Tokenization
   ->
Vocabulary
   ->
Numerical Representation
   ->
Search / Analysis
```

Preprocessing is not just a technical step. It is a design decision that shapes the behavior of the whole NLP system.
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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    display = print

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

# Download required NLTK resources if they are not already available.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

print("Libraries imported successfully.")"""
    )
)

cells.append(
    md(
        """## Part C: Load the Dataset

The CSV file contains multiple English translations. For this entire workshop series, we will use exactly **one** translation column:

```python
text_column = "daryabadi"
```

This keeps the workshop controlled and pedagogically clean.

The core columns we need are:

- `Surah`
- `Verse`
- `daryabadi`

Each row represents one verse."""
    )
)

cells.append(
    code(
        """DATA_PATH = Path("../quran_translations.csv")
text_column = "daryabadi"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\\nColumns:")
print(df.columns.tolist())

required_columns = ["Surah", "Verse", text_column]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

work_df = df[required_columns].copy()
work_df = work_df.dropna(subset=[text_column]).reset_index(drop=True)

print("\\nWorking dataframe shape:", work_df.shape)
work_df.head()"""
    )
)

cells.append(
    md(
        """## Part D: Initial Exploration

Before preprocessing, always inspect the data.

Questions we should ask:

- How many verses do we have?
- Are there missing values?
- What do sample verses look like?
- Are punctuation and capitalization present?

This is important because NLP pipelines often fail not because the model is wrong, but because the data was misunderstood."""
    )
)

cells.append(
    code(
        """print("Number of verses:", len(work_df))
print("Missing values in text column:", work_df[text_column].isna().sum())

print("\\nSample verses:")
for i, row in work_df.head(5).iterrows():
    print(f"Surah {row['Surah']}, Verse {row['Verse']}: {row[text_column][:140]}")"""
    )
)

cells.append(
    code(
        """verse_lengths = work_df[text_column].str.split().str.len()

print("Average verse length (in words):", round(verse_lengths.mean(), 2))
print("Minimum verse length:", int(verse_lengths.min()))
print("Maximum verse length:", int(verse_lengths.max()))

plt.hist(verse_lengths, bins=30, color="#2a6f97", edgecolor="white")
plt.title("Distribution of Verse Lengths")
plt.xlabel("Number of words")
plt.ylabel("Number of verses")
plt.show()"""
    )
)

cells.append(
    md(
        """## Part E: Classical Text Preprocessing

We will now walk through the standard classical preprocessing pipeline:

1. lowercasing
2. punctuation removal
3. tokenization
4. stopword removal
5. optional stemming

Not every NLP system needs every step. In fact, modern transformer-based systems often use much lighter preprocessing. But historically, these steps were central to classical NLP because older models depended heavily on surface word forms.

We will implement each step explicitly so students can see what changes."""
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


sample_text = work_df.loc[0, text_column]
print("Original text:")
print(sample_text)

normalized = basic_normalize(sample_text)
tokens = tokenize_text(normalized)
tokens_without_stopwords = remove_stopwords(tokens)
stemmed_tokens = stem_tokens(tokens_without_stopwords)

print("\\nAfter lowercasing and punctuation removal:")
print(normalized)

print("\\nTokens:")
print(tokens)

print("\\nAfter stopword removal:")
print(tokens_without_stopwords)

print("\\nAfter optional stemming:")
print(stemmed_tokens)"""
    )
)

cells.append(
    md(
        """### Why We Remove Stopwords

Stopwords are common function words such as:

- `the`
- `is`
- `and`
- `of`

These words often appear in many documents and may contribute little to topical distinction in classical retrieval systems.

However, this is not always harmless.

For example, removing stopwords can sometimes remove meaning:

```text
"not good"
```

If `not` is removed, the meaning changes dramatically. This is one reason why preprocessing is a tradeoff, not a magical rulebook."""
    )
)

cells.append(
    code(
        """def preprocess_text(text: str, remove_stops: bool = True, apply_stemming: bool = False):
    text = basic_normalize(text)
    tokens = tokenize_text(text)

    if remove_stops:
        tokens = remove_stopwords(tokens)

    if apply_stemming:
        tokens = stem_tokens(tokens)

    return tokens


work_df["normalized_text"] = work_df[text_column].apply(basic_normalize)
work_df["tokens"] = work_df[text_column].apply(preprocess_text)
work_df["stemmed_tokens"] = work_df[text_column].apply(
    lambda text: preprocess_text(text, remove_stops=True, apply_stemming=True)
)

work_df[["Surah", "Verse", text_column, "normalized_text", "tokens", "stemmed_tokens"]].head(3)"""
    )
)

cells.append(
    md(
        """## Part F: Vocabulary Creation

Once text has been tokenized, we can build a **vocabulary**.

A vocabulary is the set of unique terms that appear in the corpus.

This is one of the most important ideas in classical NLP because Bag of Words models convert documents into vectors based on vocabulary membership.

In simple terms:

- each unique word becomes a feature
- each verse becomes a vector
- the vector records how often each vocabulary word appears
"""
    )
)

cells.append(
    code(
        """all_tokens = [token for tokens in work_df["tokens"] for token in tokens]
vocabulary = sorted(set(all_tokens))
word_counts = Counter(all_tokens)

print("Vocabulary size:", len(vocabulary))
print("\\nTop 20 most common words:")
print(word_counts.most_common(20))"""
    )
)

cells.append(
    code(
        """top_words = pd.DataFrame(word_counts.most_common(15), columns=["word", "count"])

sns.barplot(data=top_words, x="count", y="word", palette="Blues_r")
plt.title("Most Frequent Words After Preprocessing")
plt.xlabel("Count")
plt.ylabel("Word")
plt.show()"""
    )
)

cells.append(
    md(
        """## Part G: Bag of Words Theory

Bag of Words (BoW) is one of the earliest and most influential text representation methods.

The central idea is simple:

- ignore grammar
- ignore deep meaning
- ignore long-range context
- count words

It is called a "bag" of words because the model largely ignores word order. It only cares about whether words occur, and how often they occur.

### Small Example

Suppose our vocabulary is:

```text
["allah", "merciful", "punishment"]
```

Then the verse:

```text
"allah is merciful"
```

might become:

```text
[1, 1, 0]
```

and the verse:

```text
"punishment is from allah"
```

might become:

```text
[1, 0, 1]
```

This is powerful because it converts text into numbers, which allows us to use mathematical tools such as vector similarity.

### But There Is a Price

BoW vectors are usually:

- high-dimensional
- sparse
- semantically shallow

We will observe all three in practice."""
    )
)

cells.append(
    code(
        """# We join the preprocessed tokens back into a cleaned string so that
# CountVectorizer works with our explicit preprocessing choices.
work_df["bow_ready_text"] = work_df["tokens"].apply(lambda tokens: " ".join(tokens))

vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(work_df["bow_ready_text"])

print("Document-term matrix shape:", X_bow.shape)
print("Matrix type:", type(X_bow))

num_nonzero = X_bow.nnz
total_cells = X_bow.shape[0] * X_bow.shape[1]
sparsity = 1 - (num_nonzero / total_cells)

print("Non-zero entries:", num_nonzero)
print("Total cells:", total_cells)
print("Sparsity:", round(sparsity, 4))"""
    )
)

cells.append(
    code(
        """feature_names = vectorizer.get_feature_names_out()

sample_dense = pd.DataFrame(
    X_bow[:5, :15].toarray(),
    columns=feature_names[:15],
    index=[f"Verse_{i}" for i in range(5)]
)

sample_dense"""
    )
)

cells.append(
    md(
        """### Why Sparse Vectors Matter

A sparse vector is a vector that contains mostly zeros.

That is exactly what happens in Bag of Words:

- the vocabulary becomes very large
- each verse contains only a tiny fraction of the vocabulary
- therefore most cells are zero

Sparse representations are not wrong, but they have important consequences:

- they use many dimensions
- they do not naturally capture meaning
- two semantically similar verses may look unrelated if they use different words
"""
    )
)

cells.append(
    md(
        """## Part H: Exact Keyword Search

Before vector-based search, many systems relied on direct term matching.

This means:

1. take a query such as `"mercy"`
2. find verses where that exact term appears
3. return the matching rows

This is easy to understand and easy to implement. It is also historically important because many early retrieval systems were built on exact term overlap."""
    )
)

cells.append(
    code(
        """def exact_keyword_search(query: str, dataframe: pd.DataFrame, text_col: str, top_k: int = 10):
    query = query.lower().strip()
    mask = dataframe[text_col].str.lower().str.contains(rf"\\b{re.escape(query)}\\b", regex=True, na=False)
    results = dataframe.loc[mask, ["Surah", "Verse", text_col]].head(top_k).copy()
    return results


keyword_query = "mercy"
keyword_results = exact_keyword_search(keyword_query, work_df, text_column, top_k=10)

print(f"Exact keyword search results for: {keyword_query!r}")
print("Number of matches shown:", len(keyword_results))
keyword_results"""
    )
)

cells.append(
    md(
        """### Limitations of Exact Keyword Search

Exact matching fails whenever the query and the document use different surface forms.

For example:

- query uses `mercy`
- verse uses `merciful`
- query uses `kindness`
- verse uses `goodness`, `compassion`, or `benevolence`

Humans see these as related. Exact keyword search usually does not.

That is our first major historical lesson:

> Exact search is useful, but lexical matching alone is too rigid for meaningful language understanding.
"""
    )
)

cells.append(
    md(
        """## Part I: Bag of Words Similarity Search

Now we move one step beyond exact matching.

Instead of asking:

```text
Does the verse contain this exact word?
```

we ask:

```text
How similar is the query vector to each verse vector?
```

This is a major conceptual step in NLP and information retrieval.

### Retrieval Pipeline

```text
User Query
   ->
Preprocess Query
   ->
Convert Query to BoW Vector
   ->
Compare With All Verse Vectors
   ->
Rank by Cosine Similarity
```

We will use **cosine similarity**, which measures the angle between vectors. If two vectors point in similar directions, they are considered similar."""
    )
)

cells.append(
    code(
        """def vectorize_query_for_bow(query: str):
    tokens = preprocess_text(query, remove_stops=True, apply_stemming=False)
    cleaned_query = " ".join(tokens)
    query_vector = vectorizer.transform([cleaned_query])
    return cleaned_query, query_vector


def bow_similarity_search(query: str, top_k: int = 5):
    cleaned_query, query_vector = vectorize_query_for_bow(query)
    similarities = cosine_similarity(query_vector, X_bow).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = work_df.loc[top_indices, ["Surah", "Verse", text_column]].copy()
    results["score"] = similarities[top_indices]
    results["cleaned_query"] = cleaned_query
    return results[["Surah", "Verse", text_column, "score", "cleaned_query"]]


bow_query = "allah is merciful"
bow_results = bow_similarity_search(bow_query, top_k=5)

print(f"Bag of Words similarity results for: {bow_query!r}")
bow_results"""
    )
)

cells.append(
    code(
        """# Let us compare the two search styles on a query phrase.
comparison_query = "allah is merciful"

print("Exact keyword search using the whole phrase:")
exact_phrase_results = exact_keyword_search(comparison_query, work_df, text_column, top_k=5)
display(exact_phrase_results)

print("\\nBag of Words similarity search:")
display(bow_similarity_search(comparison_query, top_k=5))"""
    )
)

cells.append(
    md(
        """## Part J: Interpreting the Results

Why can Bag of Words retrieve results for a multi-word query better than exact phrase matching?

Because BoW does not require the full phrase to appear in exactly the same order. It only cares about overlapping vocabulary.

So if a verse contains words such as:

- `allah`
- `merciful`

then it can still receive a non-zero similarity score even if the exact phrase `"allah is merciful"` never appears as written.

This is an improvement over exact keyword search, but it still has serious weaknesses."""
    )
)

cells.append(
    md(
        """## Part K: Limitations of Bag of Words

This is one of the most important sections in the notebook.

Bag of Words was historically important because it gave us a workable mathematical representation of text. But its limitations are precisely what pushed NLP toward better methods.

### Limitation 1: No Semantics

BoW counts word overlap. It does **not** understand meaning.

For example:

- `mercy`
- `compassion`
- `kindness`
- `benevolence`

These words may be semantically related, but a BoW model treats them as different dimensions unless the exact surface forms overlap.

### Limitation 2: No Context

BoW mostly ignores word order and local syntax.

These two phrases contain the same words:

```text
"God loves justice"
"justice loves God"
```

But they do not mean the same thing.

### Limitation 3: Sparse Vectors

As vocabulary grows, vectors become extremely large and mostly zero.

This makes representation inefficient and still does not solve the problem of meaning.

### Limitation 4: Exact-Word Dependence

If the query uses one word and the verse uses a related but different word, similarity can be weak or absent.

This is the core reason why later methods such as **TF-IDF**, **Word2Vec**, **FastText**, and **sentence embeddings** became necessary."""
    )
)

cells.append(
    code(
        """failure_query = "kindness"

print(f"Testing a failure-oriented query: {failure_query!r}")
print("\\nExact keyword search:")
display(exact_keyword_search(failure_query, work_df, text_column, top_k=10))

print("\\nBag of Words similarity search:")
display(bow_similarity_search(failure_query, top_k=5))"""
    )
)

cells.append(
    md(
        """### Why the `"kindness"` Example Matters

If the system struggles with a query like `"kindness"`, that does not necessarily mean the corpus lacks relevant meaning.

It may mean:

- the relevant verses use a different word choice
- the model cannot connect related concepts
- the representation is lexical rather than semantic

This naturally motivates the next stages of NLP evolution:

1. **TF-IDF** improves weighting
2. **N-grams** add some local context
3. **Embeddings** move toward semantic similarity
4. **Sentence embeddings** enable full semantic search
"""
    )
)

cells.append(
    md(
        """## Part L: Mini Reflection

At this point, students should be able to articulate the following:

- preprocessing changes how text is represented
- classical NLP depends heavily on token choices
- exact search is rigid
- Bag of Words allows similarity-based retrieval
- BoW is still shallow because it matches words more than meaning

This is the key educational transition:

```text
Exact matching is too rigid.
Bag of Words is more flexible.
But both remain mostly lexical systems.
```
"""
    )
)

cells.append(
    md(
        """## Part M: Summary and Transition to Notebook 2

In this notebook, we built the first fully classical retrieval pipeline:

- data loading
- preprocessing
- tokenization
- stopword removal
- optional stemming
- vocabulary creation
- Bag of Words vectors
- exact keyword search
- cosine similarity search

This is historically meaningful, but we also saw clear limitations:

- frequent words can dominate
- all words are not equally informative
- semantic similarity is still missing

### What Comes Next?

In **Notebook 2**, we will study **TF-IDF**.

TF-IDF keeps the vector-space idea but improves it by giving less importance to very common words and more importance to discriminative words.

That notebook will answer the question:

> If Bag of Words is too naive, how can we weight terms more intelligently?

---

## Limitation-to-Next-Step Map

| Limitation observed in this notebook | Why it matters | What comes next |
|---|---|---|
| Common words can dominate counts | Not all words are equally informative | **Notebook 2: TF-IDF** |
| Word order is mostly ignored | `not good` and `very good` can look misleadingly similar | **Notebook 3: N-Grams** |
| Synonyms look unrelated | `kindness` may not retrieve verses about `mercy` | **Notebooks 4-6: Embeddings** |
| Sparse vectors scale poorly | Large vocabularies produce many mostly-zero features | **Notebooks 4-7** |
| Retrieval returns text but not grounded explanation | Modern users often want context-aware answers | **Notebook 8: RAG** |

The key lesson is not that Bag of Words is bad. The key lesson is that Bag of Words is **limited in exactly the ways that later NLP methods were designed to address**.
"""
    )
)

cells.append(
    code(
        """# Optional student exercises
# These are left as plain Python comments so that instructors can expand them in class.
#
# 1. Try exact keyword search with: "forgiveness", "punishment", "charity"
# 2. Try BoW similarity search with: "god is forgiving", "help the poor", "day of judgment"
# 3. Compare results with and without stopword removal.
# 4. Rebuild the pipeline with stemming enabled and observe what changes.
# 5. Identify one verse that should be semantically relevant but is not ranked highly by BoW.

print("Notebook 1 is complete. Review the explanations and run the search cells with your own queries.")"""
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
