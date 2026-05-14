"""One-off script to emit NLP_Workshop_01_Text_Preprocessing_and_BoW.ipynb"""
import json
import uuid
from pathlib import Path

def md(s: str):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True), "id": str(uuid.uuid4())}

def code(s: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": s.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
        "id": str(uuid.uuid4()),
    }

cells = []

cells.append(md(r"""# NLP Workshop — Notebook 1  
## Text Preprocessing and Bag of Words (BoW)

*Cursor assisted — this workshop track is drafted and maintained with Cursor for consistency and reproducibility.*

**Corpus:** English translations of the Quran (CSV)  
**Translation column (fixed for the whole series):** `daryabadi` — *Abdul Majid Daryabadi*  

---

## How this notebook fits the bigger story

Modern NLP did not appear overnight. Researchers built **simple, interpretable** tools first, noticed where they broke, and then invented richer representations. This notebook is deliberately “classical”: you will touch the same ideas people used decades ago—because they still explain **why later methods exist**.

```text
Keyword Search  →  Bag of Words  →  TF‑IDF  →  N‑grams  →  Embeddings  →  Semantic Search  →  LLM + Retrieval
```

**Notebook 1 stops at BoW + cosine retrieval.** That is enough to see the first major wall: **words are not meanings**.

---

## Learning outcomes

By the end of this lab, you should be able to:

1. Load and explore a real text dataset with **Pandas**.  
2. Explain **why preprocessing** changes downstream behavior (not just “cleaning aesthetics”).  
3. Implement a transparent preprocessing pipeline: **lower casing, punctuation handling, tokenization, stopwords, optional stemming**.  
4. Build a **vocabulary** and a **document–term matrix (DTM)** using `CountVectorizer`.  
5. Run **exact keyword search** and explain its failure modes.  
6. Run **BoW + cosine similarity** retrieval and explain why it is still shallow.  
7. Articulate **limitations of BoW** with concrete “almost-right” queries (e.g., *kindness* vs *merciful*).  

---

## Prerequisite mindset

- You are allowed to treat the Quran text as **linguistic data** for learning retrieval and representation.  
- If you publish applications, be thoughtful about **faithful context** and **respectful UX**—this series focuses on **NLP mechanics**, not theology.  
"""))

cells.append(md(r"""# Part A — What is NLP (in one grounded picture)?

**Natural Language Processing (NLP)** is the discipline of turning language into **computable structure** so algorithms can search, classify, cluster, translate, summarize, and more.

Historically, NLP had two families of ideas:

1. **Rule-based / symbolic systems** (patterns, grammars, dictionaries). Strong when rules match reality; brittle when language varies.  
2. **Statistical / data-driven systems** (counts, probabilities, vectors learned from corpora). Strong at covering variation; weaker at “deep meaning” until representations improved.

**Bag of Words** belongs to the statistical family: it is simple, fast, and interpretable—but it throws away almost all linguistic detail.

---

## Why we start with BoW (even though it is “old”)

BoW answers a pedagogical question: *What is the smallest useful step from text to numbers?*  

Once you can represent a verse as a vector of word counts, you can:

- measure overlap (shared words),
- rank documents by similarity,
- feed classical machine learning models.

BoW is also the historical baseline that makes **TF‑IDF**, **n‑grams**, and **embeddings** feel motivated rather than magical.
"""))

cells.append(md(r"""# Part B — Environment setup

This notebook uses:

| Library | Role in Notebook 1 |
|---|---|
| `pandas` | load CSV, explore tabular text |
| `numpy` | lightweight numerics (arrays) |
| `re` | punctuation handling |
| `nltk` | stopwords + optional stemming + tokenization utilities |
| `sklearn` | `CountVectorizer`, cosine similarity |
| `matplotlib` / `seaborn` | plots for distributions and top token frequencies |

**Optional:** `spaCy` is imported for a short comparison demo. If the English model is missing, the notebook falls back to NLTK tokenization (so the lab still runs end-to-end).

> **Student task:** run the next cell once. If NLTK complains about missing resources, the download lines will fetch them automatically (requires internet on first run).
"""))

cells.append(code(r"""# --- Imports ---
# We keep imports explicit so you can map each library to a purpose.

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns

# `display()` is a Jupyter convenience; keep the notebook runnable in plain Python too
try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    display = print

# spaCy is optional in this notebook (model may not be installed in every lab machine)
try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

# --- NLTK data (safe to re-run) ---
# punkt: tokenizer models
# stopwords: common function words we often remove for "content word" BoW
for pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg == "stopwords" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# Plots: inline in Jupyter; `plt.show()` still works in many plain-Python runners
try:
    get_ipython().run_line_magic("matplotlib", "inline")  # type: ignore[name-defined]
except Exception:
    pass

sns.set_theme(style="whitegrid", context="notebook")

print("Python:", sys.version.split()[0])
print("OK: imports loaded.")
"""))

cells.append(md(r"""# Part C — Configure the dataset path and the translation column

**Workshop rule:** every notebook in this series uses **one** translation column so comparisons across methods stay fair.

```python
text_column = "daryabadi"
```

We also define a **relative CSV path** that works whether you open Jupyter from the project root or from inside the `cursor/` folder.
"""))

cells.append(code(r"""# --- Workshop configuration ---
text_column = "daryabadi"  # DO NOT change across notebooks in this series

# Try a few plausible locations for the CSV
_CANDIDATES = [
    Path("quran_translations.csv"),
    Path("../quran_translations.csv"),
    Path.cwd() / "quran_translations.csv",
    Path.cwd().parent / "quran_translations.csv",
]

CSV_PATH = next((p for p in _CANDIDATES if p.exists()), None)
if CSV_PATH is None:
    raise FileNotFoundError(
        "Could not find quran_translations.csv. Place it next to this notebook or in the parent folder."
    )

print("Using CSV:", CSV_PATH.resolve())
print("Using translation column:", text_column)
"""))

cells.append(md(r"""# Part D — Load the CSV and inspect the schema

**Why this matters:** NLP is rarely “just algorithms”. Most of your time is **data work**: verifying columns, spotting missing values, understanding encoding issues, and making sure each row is one logical unit (here: one verse in one translation).
"""))

cells.append(code(r"""# --- Load ---
# keep_default_na=False avoids surprises where the text "NA" becomes a missing value

df = pd.read_csv(CSV_PATH, keep_default_na=False)

print("Shape (rows, columns):", df.shape)
print("\nColumn names:")
print(list(df.columns))

if text_column not in df.columns:
    raise KeyError(f"Missing column {text_column!r}. Found: {list(df.columns)}")

# Expected columns for this workshop series
needed = {"Surah", "Verse", text_column}
missing = sorted(list(needed - set(df.columns)))
if missing:
    raise KeyError(f"Missing expected columns: {missing}")

display(df.head(5))

print("\nBasic info:")
df.info()

print("\nMissing counts (should mostly be 0 for Surah/Verse/text):")
print(df[["Surah", "Verse", text_column]].isna().sum())
"""))

cells.append(md(r"""# Part E — Create a stable verse identifier

For retrieval demos, it helps to print **Surah:Verse** alongside the translation text.
"""))

cells.append(code(r"""df["verse_id"] = df["Surah"].astype(str) + ":" + df["Verse"].astype(str)

# Quick sanity check: duplicates should be extremely rare / none for (Surah, Verse)
_dup = df.duplicated(subset=["Surah", "Verse"]).sum()
print("Duplicate (Surah, Verse) rows:", int(_dup))

df[["verse_id", text_column]].head(3)
"""))

cells.append(md(r"""## Visualization — distribution of verse lengths (characters)

**Why plot this?** Before modeling, you should understand **how long your “documents” are**. Very short verses behave differently from long ones under cosine similarity and under any future embedding model.

This histogram uses the **raw** `daryabadi` text length (characters), not tokens.
"""))

cells.append(code(r"""raw_lens = df[text_column].astype(str).str.len()

fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(raw_lens, bins=40, kde=True, ax=ax, color="steelblue")
ax.set_title("Distribution of verse length (characters) — Daryabadi column")
ax.set_xlabel("Characters per verse")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

print("min / median / max:", int(raw_lens.min()), float(raw_lens.median()), int(raw_lens.max()))
"""))

cells.append(md(r"""# Part F — Text preprocessing: what and why?

Preprocessing is not “making text pretty”. It is **committing to a representation contract**:

- If you lowercase, you decide `Merciful` and `merciful` are the same feature.  
- If you remove stopwords, you decide `the` is usually uninformative for retrieval.  
- If you stem, you decide `mercy`, `merciful`, `mercies` should collapse (sometimes helpful, sometimes harmful).

**Key idea:** every choice can **help counting overlap** or **destroy nuance**. Good NLP engineers document the contract and test behavior on real queries.

---

## Pipeline used in this notebook

```text
raw verse string
   ↓ lowercasing
   ↓ remove punctuation (keep intra-word apostrophes cautiously)
   ↓ tokenize
   ↓ remove stopwords (English)
   ↓ optional: stem (Porter; educational default OFF)
   ↓ join tokens back to a single string for vectorizers
```

We keep both:

- `text_raw` (original column copy) for **exact keyword** demos  
- `text_processed` for **BoW** demos  
"""))

cells.append(code(r"""# --- Preprocessing utilities (readable + educational) ---

# English stopwords: high-frequency function words (a/the/is/...) often dominate counts
STOP = set(stopwords.words("english"))

# Conservative punctuation removal:
# - replace most punctuation with space
# - keep apostrophes inside contractions if present (e.g., don't -> dont after cleaning)
PUNCT_RE = re.compile(r"[^a-z0-9\s']+", re.IGNORECASE)


def preprocess_text(
    text: str,
    *,
    remove_stopwords: bool = True,
    apply_stemming: bool = False,
) -> str:
    # Return a normalized whitespace-separated token string.
    # We return a STRING (not a list) because sklearn vectorizers ingest documents as strings.
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    s = str(text).lower()
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()

    tokens = word_tokenize(s)

    # Remove tokens that are just apostrophes/quotes leftovers
    tokens = [t.strip("'") for t in tokens if t.strip("'")]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP and t != ""]

    if apply_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


# Defaults for this notebook:
USE_STEMMING = False  # set True to explore stemming effects

df["text_raw"] = df[text_column].astype(str)
df["text_processed"] = df["text_raw"].apply(
    lambda x: preprocess_text(x, remove_stopwords=True, apply_stemming=USE_STEMMING)
)

print("Example raw:")
print(df.loc[0, "text_raw"][:200], "\n")
print("Example processed:")
print(df.loc[0, "text_processed"][:200])

# Empty processed rows are rare but possible; we should know about them
n_empty = (df["text_processed"].str.len() == 0).sum()
print("\nEmpty processed verses:", int(n_empty))
"""))

cells.append(md(r"""# Part G — Vocabulary “by hand” (conceptual) vs vocabulary via `CountVectorizer`

**Vocabulary** is the set of distinct word types your model knows.

You can build it with Python sets for intuition:

```text
corpus = ["mercy on parents", "mercy from the lord"]
vocabulary might include {mercy, parents, from, lord} depending on preprocessing
```

In practice, we use `CountVectorizer` because it is fast, consistent, and integrates with similarity and ML pipelines.

**Sparse vectors:** most verses use only a tiny fraction of the global vocabulary. Storing a full dense array for every verse would waste memory. `CountVectorizer` uses a **sparse matrix** internally (you will see `nnz` / sparsity ideas in the output).
"""))

cells.append(code(r"""# --- Fit a BoW model on processed verses ---

corpus = df["text_processed"].tolist()

vectorizer = CountVectorizer(
    lowercase=False,          # we already lowercased manually
    token_pattern=r"(?u)\b\w\w+\b",  # sklearn default-ish word tokens (length>=2)
)

X = vectorizer.fit_transform(corpus)  # sparse matrix (n_docs, n_features)

feature_names = vectorizer.get_feature_names_out()
print("Number of verses (documents):", X.shape[0])
print("Vocabulary size (unique word types in this preprocessing contract):", X.shape[1])

# Sparsity intuition
nnz = X.nnz
total = X.shape[0] * X.shape[1]
print(f"Nonzero entries: {nnz:,} / {total:,}  ({nnz/total:.6%} dense if materialized)")

# Show a few vocabulary entries (sorted for stable display)
print("\nSample vocabulary tokens:")
print(list(feature_names[:25]))
"""))

cells.append(md(r"""# Part H — Inspect a single verse vector (interpretability)

BoW is interpretable in a narrow sense: you can read **which words fired** for a document.

**Important limitation:** the vector does not tell you *relationships* between words—only co-presence within the same verse bag.
"""))

cells.append(code(r"""# Pick a small Surah intro verse for readability
row_idx = 0

row = df.loc[row_idx]
v = X[row_idx]  # sparse 1 x |V|

# Extract non-zero (term_index, count)
coo = v.tocoo()
pairs = list(zip(coo.col, coo.data))
pairs_sorted = sorted(pairs, key=lambda t: (-t[1], t[0]))

print("verse_id:", row["verse_id"])
print("processed text:\n", row["text_processed"][:300], "\n")
print("Top BoW counts in this verse (token, count):")
for j, c in pairs_sorted[:25]:
    print(f"  {feature_names[j]:20s}  {int(c)}")

print("\nVector shape:", v.shape)
"""))

cells.append(md(r"""# Part I — Word frequency vector for the entire corpus (global counts)

Students often confuse:

- **per-document counts** (one verse vector)  
- **global corpus counts** (how often a token appears anywhere)

Global counts explain why raw BoW retrieval can feel “dominated by common religious vocabulary” unless you introduce weighting (that is **Notebook 2: TF‑IDF**).
"""))

cells.append(code(r"""# Sum counts across all documents (still sparse-friendly)
global_counts = np.asarray(X.sum(axis=0)).ravel()
top_idx = np.argsort(-global_counts)[:25]

print("Top 25 tokens by total count (under our preprocessing contract):")
for i in top_idx:
    print(f"{feature_names[i]:20s}  {int(global_counts[i])}")
"""))

cells.append(md(r"""## Visualization — global token frequencies (BoW vocabulary)

A bar chart makes the **long tail** obvious: a few tokens dominate total mass, while most types are rare. That pattern motivates **TF‑IDF** (Notebook 2): we want rare-but-specific words to matter more than words that appear everywhere.
"""))

cells.append(code(r"""TOP_N = 20
labels = [feature_names[i] for i in top_idx[:TOP_N]]
values = [int(global_counts[i]) for i in top_idx[:TOP_N]]

fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(x=values, y=labels, ax=ax, orient="h", palette="viridis")
ax.set_title(f"Top {TOP_N} tokens by total BoW count (after preprocessing)")
ax.set_xlabel("Total count across all verses")
plt.tight_layout()
plt.show()
"""))

cells.append(md(r"""---

# Part J — Search 1: Exact keyword search (baseline intuition)

**What it is:** return verses whose *raw* translation text matches a literal substring (or literal regex).

### Example query

```python
keyword = "mercy"
```

### Why teach this?

Exact search is fast and transparent. It is also the historical baseline: **if the author uses a different word** (e.g., *compassion* instead of *mercy*), you may get **zero results** even when the meaning is close.

That gap is exactly what motivates statistical retrieval and embeddings later.
"""))

cells.append(code(r"""keyword = "mercy"

# Case-insensitive substring search on RAW text (translation column)
mask = df["text_raw"].str.contains(re.escape(keyword), case=False, na=False)
hits = df.loc[mask, ["verse_id", "text_raw"]].head(15)

print(f"Hits for exact substring match (case-insensitive): {int(mask.sum())}")
display(hits)

# Show a limitation immediately: a synonym may not appear as the same string
syn = "compassion"
mask2 = df["text_raw"].str.contains(re.escape(syn), case=False, na=False)
print(f"\nFor comparison, hits for substring {syn!r}: {int(mask2.sum())}")
print("(Not a perfect 'synonym experiment', but it illustrates literal matching behavior.)")
"""))

cells.append(md(r"""### Side-by-side lexical counts (raw text, whole-word matches)

**Borrowed teaching pattern:** count how often **related** English words actually appear as whole words in the corpus. Students can *see* why a user who types `mercy` might miss verses that prefer `merciful`, `compassion`, or `kindness`.

> Technical note: we use **word-boundary** regex matches on **raw** strings (case-insensitive). This is still “surface statistics”, not semantics—but it builds intuition before embeddings.
"""))

cells.append(code(r"""def count_whole_word(series: pd.Series, word: str) -> int:
    pat = r"(?i)\b" + re.escape(word) + r"\b"
    return int(series.str.contains(pat, regex=True, na=False).sum())


related = ["mercy", "merciful", "mercies", "kindness", "compassion", "compassionate", "gracious", "benevolent"]
counts = {w: count_whole_word(df["text_raw"], w) for w in related}

tbl = pd.DataFrame({"word": list(counts.keys()), "verses_with_word": list(counts.values())})
tbl = tbl.sort_values("verses_with_word", ascending=False).reset_index(drop=True)
display(tbl)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=tbl, x="verses_with_word", y="word", ax=ax, palette="crest")
ax.set_title("Whole-word occurrences in raw Daryabadi text (verse-level)")
ax.set_xlabel("Number of verses containing the word")
plt.tight_layout()
plt.show()
"""))

cells.append(md(r"""### Limitations of exact keyword search (discussion prompts)

- **Surface form dependence:** `mercy` ≠ `merciful` ≠ `mercies` unless you normalize morphology.  
- **No ranking beyond occurrence:** every hit looks “equally important”.  
- **No partial credit:** if the user query is a *paraphrase*, exact match can return nothing.  

**Bridge to BoW retrieval:** instead of substring match, represent query and verses as **word-count vectors** and score similarity—even if wording differs *somewhat*.
"""))

cells.append(md(r"""---

# Part K — Search 2: BoW similarity search with cosine similarity

We will use:

1. the same `CountVectorizer` (`vectorizer`) fitted on the corpus  
2. transform the query text with **the same preprocessing contract**  
3. compute **cosine similarity** between the query vector and all verse vectors  
4. return **top‑k** verses

### Example query

```python
query = "allah is merciful"
```

### Cosine similarity (intuition)

Cosine similarity measures the angle between two vectors with non-negative entries (word counts are non-negative):

- **1** means same direction (proportional counts)  
- **0** means orthogonal (no shared vocabulary under BoW)  

It is popular because it reduces the penalty of long verses vs short verses compared to raw dot products (still not “semantic understanding”).
"""))

cells.append(code(r"""def bow_topk(query: str, k: int = 10) -> pd.DataFrame:
    # Retrieve top-k verses by cosine similarity using the fitted BoW model.
    q_processed = preprocess_text(query, remove_stopwords=True, apply_stemming=USE_STEMMING)
    q_vec = vectorizer.transform([q_processed])

    sims = cosine_similarity(q_vec, X).ravel()  # shape (n_docs,)

    top_idx = np.argsort(-sims)[:k]

    out = df.loc[top_idx, ["verse_id", "text_raw", "text_processed"]].copy()
    out.insert(0, "rank", np.arange(1, k + 1))
    out["cosine_similarity"] = sims[top_idx]
    out["query_processed"] = q_processed
    return out


query = "allah is merciful"
display(bow_topk(query, k=10))

print("\nProcessed query tokens:")
print(query, "->", preprocess_text(query))
"""))

cells.append(md(r"""---

# Part L — **Limitations of Bag of Words** (this notebook’s “punchline”)

BoW answers: *“Do these texts share words?”*  
BoW does **not** reliably answer: *“Do these texts share meaning?”*

## What BoW throws away

- **Semantics:** different words can express the same idea (`kindness` vs `mercy`).  
- **Context / syntax:** `not good` vs `very good` can look similar if stopword removal deletes `not`—a classic pathology (we explore n‑grams in Notebook 3).  
- **Sparsity:** vectors are mostly zeros; overlap may be rare even when meanings align.  
- **Exact-word dependence:** if the query and verse use different lemmas/synonyms, cosine similarity can be near zero.

## Failure example (required)

```python
query = "kindness"
```

We still retrieve *something* (because some verses contain the token `kindness`), but BoW may miss many verses that express kindness using other words (mercy, compassion, benevolence, good treatment, etc.).

That failure is the educational bridge:

- **Notebook 2:** TF‑IDF improves weighting of informative vs generic terms.  
- **Later notebooks:** dense embeddings capture **semantic neighborhoods** beyond exact token overlap.
"""))

cells.append(code(r"""query = "kindness"

results = bow_topk(query, k=10)
display(results)

# Additional diagnostic: verses containing the literal token vs not
contains_token = df["text_processed"].str.contains(r"(?i)\bkindness\b", regex=True, na=False)
print("\nVerses whose processed text contains the token 'kindness':", int(contains_token.sum()))

# Show a couple high-similarity verses even when token isn't present (rare for a single-word query)
# For single-word queries, BoW cosine often reduces to overlap structure; still useful to inspect.

# Contrast query: a phrase that might relate semantically but not share the token
query2 = "be good to your parents"
display(bow_topk(query2, k=10))
"""))

cells.append(md(r"""### Student reflection (write 5–8 sentences in your notes)

Answer:

1. For `query="kindness"`, are the top results **meaningfully** about kindness, or mainly **lexically** about kindness?  
2. Find one retrieved verse that is “about kindness” semantically but **does not** contain the word *kindness*. Did it appear in the top‑10? If not, why is that expected under BoW?  
3. Why might removing stopwords help or hurt retrieval for short queries?

---

# Part M — Optional: compare tokenizers (`NLTK` vs `spaCy`)

**Why include this?** Classical NLP pipelines depend on tokenization. Different tokenizers split punctuation and contractions differently, which changes the vocabulary and retrieval scores.

If `en_core_web_sm` is unavailable, this cell prints a clear message and stops gracefully.
"""))

cells.append(code(r"""def spacy_preprocess(text: str) -> str:
    if spacy is None:
        raise RuntimeError("spaCy is not installed in this environment.")

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(str(text).lower())

    tokens = []
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        if t.is_stop:
            continue
        if t.like_num:
            # keep numbers as tokens if you want; here we keep them like our NLTK path
            tokens.append(t.text)
            continue
        tokens.append(t.text)

    return " ".join(tokens)


example = df.loc[10, "text_raw"]
print("RAW:\n", example[:220], "\n")

print("NLTK-ish pipeline used in this notebook:\n", preprocess_text(example)[:220], "\n")

if spacy is None:
    print("spaCy not installed; skipping spaCy comparison.")
else:
    try:
        print("spaCy pipeline (stopwords removed):\n", spacy_preprocess(example)[:220])
    except Exception as e:
        print("spaCy comparison failed (often missing model). Error:", repr(e))
        print("If you want the model: `python -m spacy download en_core_web_sm`")
"""))

cells.append(md(r"""---

# Appendix — Optional stemming experiment (one knob, big effects)

If you set `USE_STEMMING = True` in the preprocessing cell and rebuild `text_processed` + refit the vectorizer, you may see:

- more matches for morphological variants  
- more false positives when stems collide (**stemming over-merging**)

This appendix is intentionally short: the goal is to notice that preprocessing is a **bias lever**.
"""))

cells.append(code(r"""# If you want to experiment: flip USE_STEMMING to True in the preprocessing cell, then rerun downstream cells.

print("Stemming flag in this notebook run:", USE_STEMMING)
if USE_STEMMING:
    demo = ["running", "runs", "mercies", "merciful"]
    stemmer = PorterStemmer()
    print({w: stemmer.stem(w) for w in demo})
"""))

cells.append(md(r"""---

## Where BoW falls short — map to the rest of the series

This table (compact “lecture slide”) connects **symptoms** you observed in code to **later notebooks**. It merges ideas common in multi-author workshop drafts into one reference.

| Symptom / limitation | Example you saw (or will try) | Next step in this series |
|---|---|---|
| Common words dominate counts | `allah`, `lord` appear everywhere | **Notebook 2 — TF‑IDF** (term weighting) |
| Word order & negation ignored | `not good` vs `very good` can look similar after stopwords | **Notebook 3 — n‑grams** |
| Synonyms treated as unrelated | `kindness` vs `mercy` | **Notebooks 4–6 — embeddings** |
| Slow similarity over huge corpora | matrix math on millions of docs | **Notebook 7 — FAISS / vector indexes** |
| Retrieval alone cannot “explain” | user wants grounded answers | **Notebook 8 — RAG + LLMs** |

BoW is not “wrong”; it is **incomplete**—and that incompleteness is why the field kept inventing new representations.

---

# End of Notebook 1 — Transition to Notebook 2

You now have two retrieval baselines:

1. **Exact substring search** — brittle, but transparent.  
2. **BoW + cosine similarity** — softer matching via shared words, but still not meaning-aware.

The next notebook introduces **TF‑IDF**, which does not solve semantics either—but it *does* address a major BoW problem: **common words dominating similarity** without being informative.

```text
Notebook 1  →  BoW counts
Notebook 2  →  TF‑IDF weights (importance, not just frequency)
```

**Approval checkpoint:** review this notebook with your instructor/peers, run all cells, and tweak one preprocessing knob (stopwords on/off, stemming on/off). Observe how retrieval changes.

When you are ready, proceed to **Notebook 2 — TF‑IDF and Ranked Retrieval**.
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

out_path = Path(__file__).parent / "NLP_Workshop_01_Text_Preprocessing_and_BoW.ipynb"
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote:", out_path)

if __name__ == "__main__":
    print("Notebook generation complete.")
