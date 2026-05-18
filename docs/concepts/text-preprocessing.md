# Text Preprocessing

Before any NLP algorithm can run, raw text must be cleaned and normalized. This article explains every step of the standard preprocessing pipeline used throughout this workshop.

---

## Why Raw Text Cannot Be Used Directly

Computers work with numbers. Text in its raw form is just a sequence of characters — meaningless to mathematical algorithms.

But before we even convert to numbers, we must **standardize** the text. Consider:

- `"Allah is Most Merciful"`
- `"allah is most merciful."`

To a human, identical. To a computer without preprocessing, completely different strings. The words `"Merciful"` and `"merciful"` would be counted as two different vocabulary items.

---

## The Standard Pipeline

```
Raw Text
   |
   v
[ Step 1: Lowercasing ]           normalize letter case
   |
   v
[ Step 2: Punctuation Removal ]   remove non-alphabetic noise
   |
   v
[ Step 3: Tokenization ]          split into individual words
   |
   v
[ Step 4: Stopword Removal ]      remove high-frequency filler words
   |
   v
[ Step 5: Stemming/Lemmatization ] reduce words to root form
   |
   v
Processed Tokens
```

---

## Step 1: Lowercasing

**What it does:** Converts all characters to lowercase.

**Why it matters:** Without lowercasing, `"God"`, `"god"`, and `"GOD"` are three separate vocabulary items.

```python
text = "In the Name of Allah, the Most Beneficent"
text = text.lower()
# "in the name of allah, the most beneficent"
```

**Caveat:** For Named Entity Recognition, case carries meaning. For BoW and search, lowercasing is almost always beneficial.

---

## Step 2: Punctuation Removal

**What it does:** Removes characters that are not letters or numbers.

**Why it matters:** `"merciful"` and `"merciful,"` should be the same token.

```python
import re
text = re.sub(r"[^a-zA-Z\s]", "", text)
# "in the name of allah  the most beneficent"
```

---

## Step 3: Tokenization

**What it does:** Splits a string into a list of individual tokens.

**Why it matters:** We need word units to count, compare, and model. A string is just characters.

We use **NLTK's `word_tokenize`** — smarter than splitting on spaces, handles contractions and edge cases correctly.

```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
# ['in', 'the', 'name', 'of', 'allah', 'the', 'most', 'beneficent']
```

---

## Step 4: Stopword Removal

**What it does:** Removes extremely common words ("the", "is", "in", "of", "and") that appear in almost every document.

**Why it matters:** If every document contains "the" thousands of times, "the" carries zero discriminative power. Removing it reduces noise.

```python
from nltk.corpus import stopwords
STOP = set(stopwords.words("english"))
tokens = [t for t in tokens if t not in STOP]
# ['name', 'allah', 'beneficent']
```

!!! warning "The Stopword and Negation Trap"
    The word **"not"** is on the standard English stopword list. This means:

    | Sentence | After stopword removal | Meaning preserved? |
    |---|---|---|
    | "this book is not bad" | "book bad" | **NO** — double negative lost |
    | "this book is not good" | "book good" | **NO** — negation lost |

    For tasks where negation matters, either keep "not" in your vocabulary or move beyond BoW entirely.

---

## Step 5: Stemming

**What it does:** Reduces words to their root/base form by chopping off suffixes.

**Why it matters:** "mercy", "merciful", "mercifully" all relate to the same concept. Without stemming, a search for "mercy" misses "merciful".

```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
tokens = [stemmer.stem(t) for t in tokens]

# "merciful"  → "merci"
# "believing" → "believ"
# "prayers"   → "prayer"
```

**Note:** Stemmer outputs are not always real English words — they are root *stems*. This is fine for matching purposes.

**Alternative — Lemmatization:** Reduces words to their dictionary base form ("running" → "run"). More accurate but slower.

---

## Complete Pipeline in Code

```python
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOP = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens
```

---

## Design Decisions

Every preprocessing choice is a trade-off:

| Decision | Benefit | Cost |
|---|---|---|
| Remove stopwords | Less noise, smaller vocabulary | Negation lost ("not good" → "good") |
| Apply stemming | Merge word forms | Lose morphological distinctions |
| Set `min_df=2` | Remove typos and noise | Rare but valid words excluded |
| Keep numbers | Preserve verse references | Adds noise for semantic search |

There is no universally correct pipeline — the right choices depend on your task.

---

→ Next: [Bag-of-Words and TF-IDF in NB01 and NB02](../notebooks/nb01.md)
