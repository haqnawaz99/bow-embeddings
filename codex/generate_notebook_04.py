from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "NLP_Workshop_04_Word2Vec_Embeddings.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
cells = []

cells.append(
    md(
        """# NLP Workshop 04: Word Embeddings with Word2Vec

Codex Assisted

This notebook continues the workshop using the same dataset and the same translation column:

```python
text_column = "daryabadi"
```

This is the first notebook where the workshop makes a major conceptual shift:

> we move from sparse lexical features toward dense semantic representations.

---

## Workshop Roadmap

```text
NB01  Text Preprocessing and Bag of Words
NB02  TF-IDF and Ranked Retrieval
NB03  N-Grams and Context
NB04  Word2Vec Embeddings   <- You are here
NB05  FastText and Subword Embeddings
NB06  Sentence Embeddings and Semantic Search
NB07  Vector Databases and FAISS
NB08  RAG and LLM-based Retrieval
```

In the previous notebooks, we counted words, weighted words, and preserved short word sequences. But all of those methods still depended heavily on exact surface forms.
"""
    )
)

cells.append(
    md(
        """## Learning Outcomes

By the end of this notebook, students should be able to:

- explain why embeddings were introduced
- distinguish sparse vectors from dense vectors
- describe the core idea of Word2Vec
- explain the difference between CBOW and Skip-Gram
- train a Word2Vec model on the selected Quran translation
- inspect semantic neighbors with `most_similar`
- visualize word vectors with PCA and t-SNE
- explain why word embeddings were a major step toward semantic NLP

## Prerequisite Mindset

Ask these questions while working through the notebook:

1. Why did lexical overlap stop being enough?
2. What does it mean for words to be close in vector space?
3. What can word embeddings do that BoW, TF-IDF, and n-grams cannot do easily?
"""
    )
)

cells.append(
    md(
        """## Part A: Why Classical Lexical Models Were Not Enough

Across the first three notebooks, we saw repeated limitations:

- exact-word dependence
- sparse vectors
- weak handling of synonymy
- limited ability to connect related meanings

For example, a lexical system may treat these as unrelated dimensions:

- `mercy`
- `compassion`
- `kindness`
- `benevolence`

Humans see semantic similarity here. Classical vectorizers usually do not.

That is the historical pressure that pushed NLP toward embeddings.

### The Core Idea of Embeddings

Instead of representing each word as a huge sparse vector of counts, we learn a **dense vector** for each word.

Those dense vectors are learned from **context**:

```text
words that occur in similar contexts
tend to receive similar vector representations
```

This was a major breakthrough because it gave NLP a way to capture meaning relationships more naturally than count-based models.

### The Distributional Hypothesis

One of the most important ideas behind embeddings is the **distributional hypothesis**:

```text
You shall know a word by the company it keeps.
```

In practical NLP terms, this means:

- words that appear in similar contexts often have related meanings
- meaning can be learned from usage patterns
- context becomes the training signal

Word2Vec is one of the most famous systems built around this idea.
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

from gensim.models import Word2Vec

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    display = print

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

print("Libraries imported successfully.")"""
    )
)

cells.append(
    md(
        """## Part B: Load the Dataset

We continue with the same Quran translation column so that students can attribute differences to the representation rather than to a change in corpus.
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
        """## Part C: Preprocessing for Word Embeddings

For Word2Vec, we still normalize text, but we should think carefully about preprocessing choices.

Unlike earlier retrieval notebooks, embeddings learn from local co-occurrence patterns. That means every token contributes to the training signal.

For this notebook, we will:

- lowercase
- remove punctuation
- tokenize
- keep stopword removal optional

Why optional?

Because words that looked unimportant in Bag of Words pipelines can still play a role in contextual learning.
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


def preprocess_text(text: str, remove_stops: bool = False):
    tokens = word_tokenize(basic_normalize(text))
    if remove_stops:
        tokens = [token for token in tokens if token not in stop_words]
    return tokens


work_df["tokens_for_embeddings"] = work_df[text_column].apply(lambda text: preprocess_text(text, remove_stops=False))
work_df[[text_column, "tokens_for_embeddings"]].head(3)"""
    )
)

cells.append(
    md(
        """## Part D: Sparse vs Dense Representations

Before training Word2Vec, let us make the contrast explicit.

### Sparse Representation

In earlier notebooks, a word or verse could be represented in a very large vocabulary space:

- many dimensions
- mostly zeros
- no natural notion of semantic closeness

### Dense Representation

In embeddings:

- vectors have relatively few dimensions, such as 50, 100, or 300
- almost every dimension has a real-valued number
- semantic similarity can emerge through geometry

This is one of the most important conceptual transitions in NLP.
"""
    )
)

cells.append(
    code(
        """demo_vocab = ["mercy", "compassion", "punishment", "guidance", "believers"]
word_to_index = {word: i for i, word in enumerate(demo_vocab)}


def one_hot(word, vocab_size, mapping):
    vector = np.zeros(vocab_size)
    if word in mapping:
        vector[mapping[word]] = 1
    return vector


mercy_sparse = one_hot("mercy", len(demo_vocab), word_to_index)
compassion_sparse = one_hot("compassion", len(demo_vocab), word_to_index)

sparse_cosine = np.dot(mercy_sparse, compassion_sparse) / (
    (np.linalg.norm(mercy_sparse) * np.linalg.norm(compassion_sparse)) + 1e-10
)

print("Sparse one-hot vector for 'mercy':")
print(mercy_sparse)
print("\\nSparse one-hot vector for 'compassion':")
print(compassion_sparse)
print("\\nCosine similarity between one-hot vectors:", sparse_cosine)
print("\\nObservation: in a sparse lexical identity space, different words are entirely unrelated unless they are literally the same token.")"""
    )
)

cells.append(
    md(
        """## Part E: What Is Word2Vec?

**Word2Vec** is a family of neural methods for learning word embeddings from context.

It became influential because it offered a practical way to learn meaningful dense word vectors from unlabeled text.

### Two Main Training Styles

#### 1. CBOW (Continuous Bag of Words)

CBOW tries to predict a target word from its surrounding context words.

```text
context words  ->  target word
```

#### 2. Skip-Gram

Skip-Gram does the opposite. It tries to predict surrounding words from a target word.

```text
target word  ->  surrounding context words
```

### Intuition

If two words often appear in similar contexts, the model learns similar vectors for them.

This is why embeddings are such a major step beyond BoW and TF-IDF.
"""
    )
)

cells.append(
    md(
        """## Part F: Prepare the Training Corpus

Word2Vec expects a corpus as a list of tokenized sentences.

In this workshop, each verse will act as a small sentence-like training unit.
"""
    )
)

cells.append(
    code(
        """sentences = work_df["tokens_for_embeddings"].tolist()

print("Number of training sentences:", len(sentences))
print("First tokenized verse:")
print(sentences[0])

sentence_lengths = [len(s) for s in sentences]
print("\\nAverage tokenized verse length:", round(np.mean(sentence_lengths), 2))"""
    )
)

cells.append(
    md(
        """## Part G: Train a Word2Vec Model

We will train a small educational Word2Vec model.

Important parameters:

- `vector_size`: the dimensionality of the embedding space
- `window`: how many neighboring words the model looks at
- `min_count`: minimum frequency for a word to be included
- `sg`: `0` for CBOW, `1` for Skip-Gram

For a small workshop corpus, students should expect a modest model. The goal here is understanding, not state-of-the-art performance.
"""
    )
)

cells.append(
    code(
        """word2vec_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=1,
    sg=1,  # 1 = Skip-Gram, 0 = CBOW
    epochs=20,
    seed=42,
)

print("Word2Vec model trained successfully.")
print("Vocabulary size:", len(word2vec_model.wv))"""
    )
)

cells.append(
    md(
        """## Part H: Inspect the Learned Vocabulary

Not every token will be present in the final vocabulary.

That is because `min_count=2` excludes very rare words. This is useful because extremely rare words often do not provide stable training signals in small corpora.
"""
    )
)

cells.append(
    code(
        """vocab_preview = list(word2vec_model.wv.index_to_key[:30])
print("First 30 words in the learned vocabulary:")
print(vocab_preview)"""
    )
)

cells.append(
    md(
        """## Part I: Explore Semantic Neighbors

This is one of the most exciting parts of the notebook.

With count-based lexical models, asking for the "closest" words in meaning is awkward. With embeddings, it becomes natural.

We can ask questions like:

```python
model.wv.most_similar("mercy")
```

If the corpus and model are informative enough, the returned words may reflect semantic or contextual similarity.
"""
    )
)

cells.append(
    code(
        """query_word = "mercy"

if query_word in word2vec_model.wv:
    similar_words = word2vec_model.wv.most_similar(query_word, topn=10)
    similar_df = pd.DataFrame(similar_words, columns=["word", "similarity"])
    print(f"Most similar words to {query_word!r}:")
    display(similar_df)
else:
    print(f"{query_word!r} is not in the learned vocabulary. Try a more frequent word.")"""
    )
)

cells.append(
    code(
        """candidate_words = ["mercy", "merciful", "allah", "believers", "punishment", "guidance"]

for word in candidate_words:
    print("=" * 80)
    print(f"WORD: {word}")
    if word in word2vec_model.wv:
        display(pd.DataFrame(word2vec_model.wv.most_similar(word, topn=5), columns=["word", "similarity"]))
    else:
        print("Not in vocabulary.")"""
    )
)

cells.append(
    md(
        """## Part J: Inspect Actual Dense Vectors

Unlike sparse count vectors, embeddings are dense numerical arrays.

Each dimension does not usually have a simple human-readable label. The meaning emerges from the whole vector and its relationship to other vectors.
"""
    )
)

cells.append(
    code(
        """sample_word = "allah" if "allah" in word2vec_model.wv else word2vec_model.wv.index_to_key[0]
sample_vector = word2vec_model.wv[sample_word]

print(f"Sample word: {sample_word}")
print("Vector shape:", sample_vector.shape)
print("First 15 dimensions:")
print(sample_vector[:15])"""
    )
)

cells.append(
    md(
        """## Part K: Cosine Similarity Between Words

Word2Vec vectors live in a geometric space. Words that are contextually similar often have higher cosine similarity.

This is one reason embeddings became so influential: similarity is no longer based only on exact overlap, but on learned representation.
"""
    )
)

cells.append(
    code(
        """def cosine_sim(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


pairs = [
    ("mercy", "merciful"),
    ("guidance", "believers"),
    ("punishment", "mercy"),
]

rows = []
for w1, w2 in pairs:
    if w1 in word2vec_model.wv and w2 in word2vec_model.wv:
        rows.append(
            {
                "word_1": w1,
                "word_2": w2,
                "cosine_similarity": cosine_sim(word2vec_model.wv[w1], word2vec_model.wv[w2]),
            }
        )

pd.DataFrame(rows)"""
    )
)

cells.append(
    md(
        """## Part L: PCA Visualization

Dense embeddings are hard to interpret directly because each vector may have 100 dimensions or more.

One way to explore them is to reduce them to 2 dimensions for visualization.

**PCA** is a linear dimensionality reduction technique. It helps us project the vectors into a smaller space while preserving as much variance as possible.
"""
    )
)

cells.append(
    code(
        """plot_words = [word for word in ["allah", "mercy", "merciful", "guidance", "believers", "punishment", "forgiveness", "truth", "path", "lord"] if word in word2vec_model.wv]

word_vectors = np.array([word2vec_model.wv[word] for word in plot_words])

pca = PCA(n_components=2, random_state=42)
word_vectors_2d = pca.fit_transform(word_vectors)

pca_df = pd.DataFrame(word_vectors_2d, columns=["PC1", "PC2"])
pca_df["word"] = plot_words

plt.figure(figsize=(10, 7))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", s=100)

for _, row in pca_df.iterrows():
    plt.text(row["PC1"] + 0.01, row["PC2"] + 0.01, row["word"], fontsize=10)

plt.title("PCA Projection of Selected Word2Vec Embeddings")
plt.show()"""
    )
)

cells.append(
    md(
        """## Part M: t-SNE Visualization

**t-SNE** is another dimensionality reduction method. It is often better than PCA for visualizing local neighborhood structure, although it can be more sensitive to parameters.

For teaching purposes:

- PCA is easier to explain
- t-SNE often produces visually clearer clusters

Students should understand that these plots are only approximations of the original high-dimensional geometry.
"""
    )
)

cells.append(
    code(
        """if len(plot_words) >= 5:
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(plot_words) - 1), max_iter=1000)
    tsne_vectors_2d = tsne.fit_transform(word_vectors)

    tsne_df = pd.DataFrame(tsne_vectors_2d, columns=["x", "y"])
    tsne_df["word"] = plot_words

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=tsne_df, x="x", y="y", s=100, color="darkgreen")

    for _, row in tsne_df.iterrows():
        plt.text(row["x"] + 0.3, row["y"] + 0.3, row["word"], fontsize=10)

    plt.title("t-SNE Projection of Selected Word2Vec Embeddings")
    plt.show()
else:
    print("Not enough words available for a stable t-SNE visualization.")"""
    )
)

cells.append(
    md(
        """## Part N: What Word2Vec Improves

Word2Vec is historically important because it improves on earlier methods in several ways:

- words get dense vectors instead of sparse count features
- contextual similarity can emerge naturally
- semantically related words may become neighbors
- vector spaces become more flexible for downstream tasks

This is one of the first major moments in NLP where **meaning starts to emerge from usage patterns** rather than from direct word overlap alone.
"""
    )
)

cells.append(
    md(
        """## Part O: Limitations of Word2Vec

Word2Vec was a major advance, but it is not the endpoint of NLP.

### Limitation 1: Word-Level Only

It learns vectors for individual words, not full sentences.

### Limitation 2: One Vector per Word Type

A word usually gets one embedding regardless of its different possible meanings.

This is why Word2Vec is often called a **static embedding** model:

- one word type
- one learned vector
- multiple senses collapsed together

### Limitation 3: Rare Words Remain Difficult

Rare and morphologically complex words may still be hard to learn well.

### Limitation 4: OOV (Out-of-Vocabulary) Words

If a word was never learned properly, or does not appear in the final vocabulary, Word2Vec cannot provide a vector for it.

This is especially important for:

- rare words
- spelling variants
- morphologically rich forms
- unseen tokens at inference time

### Limitation 5: No Full Retrieval Pipeline Yet

Word vectors alone do not automatically solve verse-level search in the modern semantic sense.

These limitations motivate the next steps:

- FastText for subword information
- sentence embeddings for full semantic retrieval
"""
    )
)

cells.append(
    md(
        """## Part P: Discussion Prompts

1. Why is a dense vector representation more useful than a sparse count vector for semantic similarity?
2. Why might `mercy` and `merciful` end up close in Word2Vec space?
3. Why is Word2Vec still not enough for full sentence-level search?
4. Why might small training corpora produce weaker embeddings than very large corpora?
"""
    )
)

cells.append(
    md(
        """## Part Q: Summary and Transition to Notebook 5

In this notebook, we made the workshop's first major move into semantic representation.

### What We Learned

- sparse lexical vectors are limited for meaning
- Word2Vec learns dense vectors from context
- similar contexts can produce similar embeddings
- `most_similar` gives a practical view into semantic neighborhoods
- PCA and t-SNE help visualize high-dimensional embedding spaces

### Historical Insight

This notebook marks a major transition in NLP history:

```text
from counting words
to learning word meanings from context
```

### What Comes Next?

Notebook 5 introduces **FastText**.

That notebook asks a new question:

> what if words should be represented not only as whole tokens, but also through their subword pieces?

---

## Limitation-to-Next-Step Map

| What Word2Vec improves | What still remains weak | What comes next |
|---|---|---|
| Dense semantic word vectors | Rare words and morphology | **Notebook 5: FastText** |
| Context-based word similarity | Sentence-level meaning | **Notebook 6: Sentence Embeddings** |
| Less exact-word dependence | One vector per word type | **More advanced embeddings** |
| Better semantic neighborhoods | Full retrieval still incomplete | **Semantic search systems** |
"""
    )
)

cells.append(
    md(
        """## Mini Self-Check

- [ ] I can explain why embeddings were introduced.
- [ ] I understand the difference between sparse and dense vectors.
- [ ] I can explain the main idea of Word2Vec.
- [ ] I can interpret `most_similar` results carefully.
- [ ] I understand why Word2Vec motivates FastText and sentence embeddings.
"""
    )
)

cells.append(
    code(
        """# Suggested student exercises
# 1. Try `most_similar` on several frequent words and compare the neighborhoods.
# 2. Increase or decrease `window` and observe how the neighborhoods change.
# 3. Train a CBOW model by changing `sg=0` and compare a few results.
# 4. Try a larger or smaller `vector_size` and observe the effect on visualization.
# 5. Identify one case where the learned similarity seems meaningful and one where it seems noisy.

print("Notebook 4 is complete. Review the semantic neighborhoods before moving to FastText.")"""
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
