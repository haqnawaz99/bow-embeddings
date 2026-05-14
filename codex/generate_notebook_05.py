from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "NLP_Workshop_05_FastText_and_Subword_Embeddings.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
cells = []

cells.append(
    md(
        """# NLP Workshop 05: FastText and Subword Embeddings

Codex Assisted

This notebook continues the workshop using the same dataset and the same translation column:

```python
text_column = "daryabadi"
```

This notebook asks a very important question:

> what should we do when a word is rare, unseen, or morphologically complex?

Word2Vec was a major step forward, but it still treated each word mostly as a whole unit. FastText moves beyond that by learning from **subword pieces**.

---

## Workshop Roadmap

```text
NB01  Text Preprocessing and Bag of Words
NB02  TF-IDF and Ranked Retrieval
NB03  N-Grams and Context
NB04  Word2Vec Embeddings
NB05  FastText and Subword Embeddings   <- You are here
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

- explain why Word2Vec struggles with rare and unseen words
- describe the core idea of FastText
- understand character n-grams and subword modeling
- explain why FastText is important for morphologically rich languages
- train a FastText model on the selected Quran translation
- compare FastText and Word2Vec on word similarity tasks
- explain why subword modeling helps but still does not solve sentence meaning

## Prerequisite Mindset

Keep asking:

1. What happens when a whole-word model sees a rare word?
2. Can internal word structure carry useful information?
3. Why would subword modeling matter even more in Arabic, Urdu, or other morphologically rich languages?
"""
    )
)

cells.append(
    md(
        """## Part A: Why Word2Vec Was Still Not Enough

In Notebook 4, we learned dense word embeddings with Word2Vec. That was a major advance, but several problems remained:

- rare words may be poorly learned
- unseen words may be missing entirely
- morphology is mostly ignored
- spelling variation can be difficult

For example, words like:

- `mercy`
- `merciful`
- `mercifully`

are related in form and meaning. A whole-word model may not make strong use of that internal structure.

FastText addresses this by representing words through **character n-grams** as well as whole-word identity.
"""
    )
)

cells.append(
    md(
        """## Part B: Why This Matters for Arabic and Urdu

FastText is especially important for morphologically rich languages.

In languages like Arabic and Urdu:

- words can take many related surface forms
- prefixes, suffixes, and inflections carry meaning
- rare forms appear often enough to matter

Even though our workshop corpus here is an English translation, this notebook is a good place to explain why subword modeling became historically important for multilingual and morphologically rich NLP.

### Key Lesson

If meaning is partly stored inside word structure, then learning from whole tokens alone may be too weak.
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

from gensim.models import Word2Vec, FastText

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
        """## Part C: Load the Dataset

We keep the same dataset and the same translation column so the representation change remains the main difference.
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
        """## Part D: Preprocessing for Subword Embeddings

FastText learns from token sequences, but also from character-level subpieces.

We will use a light preprocessing pipeline:

- lowercase
- remove punctuation
- tokenize

We will not remove stopwords by default, because the model learns from contextual usage patterns.
"""
    )
)

cells.append(
    code(
        """def basic_normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def preprocess_text(text: str):
    return word_tokenize(basic_normalize(text))


work_df["tokens_for_embeddings"] = work_df[text_column].apply(preprocess_text)
work_df[[text_column, "tokens_for_embeddings"]].head(3)"""
    )
)

cells.append(
    md(
        """## Part E: What Is FastText?

FastText is an extension of Word2Vec introduced by Facebook AI Research.

Its key idea is:

> represent a word not only as a whole token, but also through its character n-grams.

For example, the word:

```text
merciful
```

might be represented partly through pieces like:

```text
mer
erc
rci
cif
ifu
ful
```

plus longer overlapping character fragments.

This means related word forms can share subword information.
"""
    )
)

cells.append(
    md(
        """## Part F: Character N-Grams Intuition

Character n-grams are short overlapping sequences of characters.

### Example

For the word:

```text
guidance
```

some character 3-grams are:

```text
gui
uid
ida
dan
anc
nce
```

These fragments help the model connect words that share internal structure.

In many FastText explanations, words are conceptually wrapped with boundary markers such as:

```text
<guidance>
```

That helps the model distinguish:

- prefixes
- suffixes
- internal fragments

So subword modeling is not only about pieces, but also about where those pieces occur inside the word.
"""
    )
)

cells.append(
    code(
        """def character_ngrams(word: str, n: int = 3):
    return [word[i:i+n] for i in range(len(word) - n + 1)]


example_word = "guidance"
print("Word:", example_word)
print("Character 3-grams:")
print(character_ngrams(example_word, n=3))"""
    )
)

cells.append(
    md(
        """## Part G: Prepare the Training Corpus

Both Word2Vec and FastText expect a corpus as a list of tokenized sentences.

We will reuse the same verse-level tokenization for both models so the comparison is fair.
"""
    )
)

cells.append(
    code(
        """sentences = work_df["tokens_for_embeddings"].tolist()

print("Number of training sentences:", len(sentences))
print("First tokenized verse:")
print(sentences[0])"""
    )
)

cells.append(
    md(
        """## Part H: Train a Baseline Word2Vec Model

We train Word2Vec again so that we can compare its behavior directly with FastText inside the same notebook.
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
    sg=1,
    epochs=20,
    seed=42,
)

print("Word2Vec model trained.")
print("Word2Vec vocabulary size:", len(word2vec_model.wv))"""
    )
)

cells.append(
    md(
        """## Part I: Train a FastText Model

Now we train FastText with broadly comparable settings.

The main conceptual difference is that FastText learns word vectors with access to subword information.
"""
    )
)

cells.append(
    code(
        """fasttext_model = FastText(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=1,
    sg=1,
    epochs=20,
    seed=42,
)

print("FastText model trained.")
print("FastText vocabulary size:", len(fasttext_model.wv))"""
    )
)

cells.append(
    md(
        """## Part J: Compare Word Similarity Results

This is the main practical comparison.

We will query both models and compare their nearest neighbors.

Students should watch for:

- whether similar forms cluster together
- whether semantically related words appear
- whether FastText behaves better on morphologically related forms
"""
    )
)

cells.append(
    code(
        """def show_similar(model, word: str, model_name: str, topn: int = 8):
    print("=" * 80)
    print(f"{model_name} -> most similar words for: {word!r}")
    if word in model.wv:
        display(pd.DataFrame(model.wv.most_similar(word, topn=topn), columns=["word", "similarity"]))
    else:
        print("Word is not available in the vocabulary.")


comparison_words = ["mercy", "merciful", "guidance", "believers", "punishment"]

for word in comparison_words:
    show_similar(word2vec_model, word, "Word2Vec")
    show_similar(fasttext_model, word, "FastText")"""
    )
)

cells.append(
    md(
        """## Part K: Compare Morphologically Related Words

This is where FastText often shows its strength.

If two words share meaningful internal structure, FastText may connect them more naturally than a whole-word model.
"""
    )
)

cells.append(
    code(
        """morphology_pairs = [
    ("mercy", "merciful"),
    ("guide", "guidance"),
    ("believe", "believers"),
]

rows = []
for w1, w2 in morphology_pairs:
    row = {"word_1": w1, "word_2": w2}

    if w1 in word2vec_model.wv and w2 in word2vec_model.wv:
        row["word2vec_similarity"] = word2vec_model.wv.similarity(w1, w2)
    else:
        row["word2vec_similarity"] = np.nan

    # FastText can usually produce vectors even when a form is rare if subword info helps.
    row["fasttext_similarity"] = fasttext_model.wv.similarity(w1, w2)
    rows.append(row)

pd.DataFrame(rows)"""
    )
)

cells.append(
    md(
        """## Part L: Rare and Unseen Word Behavior

One of the biggest historical motivations for FastText is this:

Word2Vec often struggles when a word is rare or unseen.

FastText can still build a vector from character n-grams, even if the exact word is weakly represented.

This does not make FastText magically perfect, but it often makes it more robust.
"""
    )
)

cells.append(
    code(
        """oov_probe = "mercifully"

print("Attempting direct Word2Vec lookup for an OOV-like word:")
try:
    _ = word2vec_model.wv[oov_probe]
    print("Word2Vec vector found.")
except KeyError as exc:
    print("Word2Vec lookup failed with KeyError-style behavior:")
    print(exc)

print("\\nFastText direct lookup for the same word:")
fasttext_vector = fasttext_model.wv[oov_probe]
print("FastText returned a vector with shape:", fasttext_vector.shape)"""
    )
)

cells.append(
    code(
        """test_words = ["mercifully", "guiding", "believer", "unseenwordexample"]

rows = []
for word in test_words:
    rows.append(
        {
            "word": word,
            "in_word2vec_vocab": word in word2vec_model.wv.key_to_index,
            "in_fasttext_vocab": word in fasttext_model.wv.key_to_index,
            "fasttext_vector_available": True,  # FastText can synthesize vectors from subwords
        }
    )

pd.DataFrame(rows)"""
    )
)

cells.append(
    code(
        """probe_word = "mercifully"

print("Word2Vec behavior:")
if probe_word in word2vec_model.wv:
    display(pd.DataFrame(word2vec_model.wv.most_similar(probe_word, topn=5), columns=["word", "similarity"]))
else:
    print(f"{probe_word!r} is not available as a learned whole-word vector.")

print("\\nFastText behavior:")
display(pd.DataFrame(fasttext_model.wv.most_similar(probe_word, topn=5), columns=["word", "similarity"]))"""
    )
)

cells.append(
    md(
        """## Part M: Typo Robustness Demonstration

Subword modeling can sometimes help even when a token is misspelled or slightly varied.

This does not mean FastText fully understands spelling noise, but it often behaves more gracefully than a pure whole-word lookup table.
"""
    )
)

cells.append(
    code(
        """typo_word = "mercifull"

print("Word2Vec behavior on typo-like input:")
try:
    display(pd.DataFrame(word2vec_model.wv.most_similar(typo_word, topn=5), columns=["word", "similarity"]))
except KeyError as exc:
    print("Word2Vec failed to look up the typo-like word.")
    print(exc)

print("\\nFastText behavior on typo-like input:")
display(pd.DataFrame(fasttext_model.wv.most_similar(typo_word, topn=5), columns=["word", "similarity"]))"""
    )
)

cells.append(
    md(
        """## Part N: Why FastText Helps

FastText improves on Word2Vec because it can:

- share information across related word forms
- produce more robust vectors for rare words
- handle some unseen words through subword composition
- make morphology matter in the representation

This is especially useful in real-world NLP where text is noisy, varied, and morphologically rich.
"""
    )
)

cells.append(
    md(
        """## Part O: What FastText Still Does Not Solve

FastText is stronger than Word2Vec in several ways, but it still has limits.

### Limitation 1: Still a Word-Level Model

It improves word representations, but it does not directly represent full sentence meaning.

### Limitation 2: Still Mostly Static

Like Word2Vec, it still tends to assign one main vector per word type, rather than dynamically adapting fully to sentence context.

### Limitation 3: Subword Similarity Is Not Always Semantic Similarity

Words can look similar in form without actually being close in meaning.

### Limitation 4: Hashing and Bucket Approximation

FastText typically stores subword information through a hashing-based bucket mechanism.

That is efficient, but it introduces approximation:

- different subword fragments can sometimes collide
- the model gains efficiency, but not perfect symbolic precision

### Limitation 5: Full Semantic Retrieval Still Needs Sentence-Level Methods

For modern semantic search systems, we usually want sentence embeddings rather than only word embeddings.
"""
    )
)

cells.append(
    md(
        """## Part P: Discussion Prompts

1. Why might FastText be more robust than Word2Vec for rare words?
2. Why is FastText especially relevant for Arabic and Urdu?
3. Can character similarity ever mislead the model?
4. Why do we still need sentence embeddings after FastText?
"""
    )
)

cells.append(
    md(
        """## Part Q: Summary and Transition to Notebook 6

In this notebook, we improved word embeddings by adding subword awareness.

### What We Learned

- Word2Vec treats words mostly as whole units
- FastText uses character n-grams
- subword modeling helps with morphology, rare words, and some unseen words
- FastText is especially useful for morphologically rich languages
- FastText is still not enough for full sentence-level semantic search

### Historical Insight

This notebook shows another major step in NLP evolution:

```text
from whole-word embeddings
to subword-aware embeddings
```

### What Comes Next?

Notebook 6 introduces **sentence embeddings and semantic search**.

That is the next major leap:

> from learning individual word meaning to representing the meaning of whole queries and whole verses.

---

## Limitation-to-Next-Step Map

| What FastText improves | What still remains weak | What comes next |
|---|---|---|
| Rare word handling | Full sentence meaning | **Notebook 6: Sentence Embeddings** |
| Morphological robustness | Query-to-verse semantic matching | **Semantic search** |
| Subword-based representations | Deep context understanding | **Sentence-level models** |
| Better word-level coverage | Whole-document retrieval | **Dense retrieval systems** |
"""
    )
)

cells.append(
    md(
        """## Mini Self-Check

- [ ] I can explain how FastText differs from Word2Vec.
- [ ] I understand character n-grams and subword modeling.
- [ ] I know why FastText helps with rare or unseen words.
- [ ] I understand why FastText matters for Arabic and Urdu.
- [ ] I understand why sentence embeddings are still needed next.
"""
    )
)

cells.append(
    code(
        """# Suggested student exercises
# 1. Compare Word2Vec and FastText neighbors for several related words.
# 2. Try a few rare-looking word forms and observe which model is more robust.
# 3. Change FastText vector_size or window and see how neighborhoods change.
# 4. Identify one case where subword similarity helps and one where it may mislead.

print("Notebook 5 is complete. Review the FastText comparisons before moving to sentence embeddings.")"""
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
