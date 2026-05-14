"""Emit NLP_Workshop_05_FastText_and_Subword_Embeddings.ipynb — FastText vs Word2Vec, OOV vectors."""
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


cells: list = []

cells.append(
    md(
        r"""# NLP Workshop — Notebook 5  
## FastText and Subword Embeddings

*Cursor assisted — this workshop track is drafted and maintained with Cursor for consistency and reproducibility.*

**Corpus:** English translations of the Quran (CSV)  
**Translation column:** `daryabadi`

---

## Where you are in the roadmap

```text
NB01  Preprocess + BoW  →  NB02  TF‑IDF  →  NB03  N‑grams  →  NB04  Word2Vec  →  NB05  FastText  →  NB06+  Sentences / FAISS / RAG
                                                                                              ▲
                                                                                       YOU ARE HERE
```

**Notebook 4** trained **Word2Vec**: one vector per **word type** in vocabulary. Any token that never met `min_count` (or a typo / rare morph) is effectively **out of vocabulary (OOV)** for lookup.

**FastText** (Bojanowski et al., 2017) keeps the same Skip‑gram / CBOW training loop but represents each word using **character n‑grams** (substrings). Rare and unseen words can still get a reasonable vector if their **pieces** appeared in training.

---

## Learning outcomes

1. Explain the **OOV / rare word** problem for Word2Vec-style lookup tables.  
2. Describe **FastText’s subword** idea at a high level (character n‑grams summed into a word vector).  
3. Train **FastText** on the same verse token lists as Notebook 4.  
4. Compare **`most_similar`** behavior for an in‑vocabulary anchor (e.g. **`mercy`**) against a **Skip‑gram Word2Vec** model trained on the same data.  
5. Show that **FastText can produce a vector for an OOV string** and run **`most_similar`** from it; contrast with **Word2Vec `KeyError`**.  
6. Compare **morphological neighbors** (`mercy` / `merciful` / …) and a **typo** string with both models.  
7. Run a **lightweight `min_n` / `max_n` sensitivity** study (smaller subsample + fewer epochs) and interpret neighbor drift.  
8. Reuse **mean‑pooled verse vectors** to compare **verse retrieval** (Word2Vec skips unknown tokens vs FastText composes per token).  
9. Read a **2D PCA** plot of the **same word set** under Word2Vec vs FastText.  
10. Name limitations that still motivate **sentence embeddings (NB06)**.

**Prerequisites:** Notebooks 1–4.

---
"""
    )
)

cells.append(
    md(
        r"""# Part A — What problem does FastText try to fix?

## Word2Vec recap (one line)

Word2Vec learns a **row in a matrix** for each frequent word type. Lookup is fast and interpretable—but **closed vocabulary**.

## Failure modes you have already felt

- **Rare words** get dropped by `min_count` or have unstable vectors.  
- **Typos and morphological variants** (`mercy` vs `mercies` vs `merciful`) do not automatically share one row.  
- **Out-of-vocabulary tokens** at inference: no row → **no vector** (typically `KeyError` in `gensim`).

## FastText’s move

Instead of *only* learning a whole-word vector, FastText learns vectors for **character n‑grams** (within each word) and **combines** them to form the word representation.

**Pedagogical payoff:** the model can **generalize** to strings whose **substrings** were seen often, even if the exact word type was rare or unseen.

---
"""
    )
)

cells.append(
    md(
        r"""# Part B — Subword n‑grams (intuition, not a full proof)

Think of a word as a bag of **short character pieces** (length `min_n` … `max_n`), plus the whole word as a special token.

**Example sketch (not exact `gensim` delimiter rules):**

- For `mercy`, character trigrams might include pieces like `mer`, `erc`, `rcy` (illustrative).  
- A novel string `mercifully` shares many pieces with words you *did* train on → its aggregate vector can land “near” mercy‑related geometry.

**Hyperparameters you will see in code:**

| Parameter | Role |
|---|---|
| `min_n`, `max_n` | Shortest / longest character n‑gram lengths used for subwords |
| `bucket` | Hash buckets for rare n‑grams (memory / collision tradeoff) |

`gensim.models.FastText` implements training + inference compatible with the rest of the workshop stack.

---

### Why this matters beyond “English Daryabadi”

Morphologically rich languages (e.g. **Arabic**, **Urdu**) have many **surface forms** per lemma. Subword models are widely used in multilingual pipelines partly because **shared character pieces** carry signal when whole-word counts are thin. This notebook’s corpus is English, but the **mechanism** you train here is the same one deployed for those languages.

---
"""
    )
)

cells.append(
    md(
        r"""# Part C — Imports, data, token sentences (same contract as NB04)

We reuse **`preprocess_text`** and the same **verse-aligned** token lists so results are comparable to Notebook 4.
"""
    )
)

cells.append(
    code(
        r"""from __future__ import annotations

import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim.models import FastText, Word2Vec

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    display = print

warnings.filterwarnings("ignore", category=UserWarning)

for pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find("corpora/stopwords" if pkg == "stopwords" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

try:
    get_ipython().run_line_magic("matplotlib", "inline")  # type: ignore[name-defined]
except Exception:
    pass

sns.set_theme(style="whitegrid", context="notebook")
print("OK: imports loaded.")
"""
    )
)

cells.append(
    code(
        r"""text_column = "daryabadi"

_CANDIDATES = [
    Path("quran_translations.csv"),
    Path("../quran_translations.csv"),
    Path.cwd() / "quran_translations.csv",
    Path.cwd().parent / "quran_translations.csv",
]

CSV_PATH = next((p for p in _CANDIDATES if p.exists()), None)
if CSV_PATH is None:
    raise FileNotFoundError("Could not find quran_translations.csv.")

df = pd.read_csv(CSV_PATH, keep_default_na=False)
df["text_raw"] = df[text_column].astype(str)

STOP = set(stopwords.words("english"))
PUNCT_RE = re.compile(r"[^a-z0-9\s']+", re.IGNORECASE)


def preprocess_text(text: str) -> str:
    if not text:
        return ""
    s = str(text).lower()
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = word_tokenize(s)
    tokens = [t.strip("'") for t in tokens if t.strip("'")]
    tokens = [t for t in tokens if t not in STOP and t != ""]
    return " ".join(tokens)


verse_texts = []
verse_ids = []
sentences = []
for _, row in df.iterrows():
    raw = str(row["text_raw"])
    s = preprocess_text(raw)
    toks = s.split()
    if len(toks) >= 3:
        verse_texts.append(raw)
        verse_ids.append(f"{row['Surah']}:{row['Verse']}")
        sentences.append(toks)

print("Verses with >=3 tokens after preprocess:", len(sentences))
print("Example verse_id:", verse_ids[0])
print("Example sentence (first 15 tokens):", sentences[0][:15])
"""
    )
)

cells.append(
    md(
        r"""# Part D — Train Skip‑gram Word2Vec vs FastText (matched settings)

We keep **`vector_size`**, **`window`**, **`min_count`**, **`epochs`**, and **`seed`** aligned so differences come mostly from **subword modeling** (FastText) vs **whole-word rows only** (Word2Vec).

Training is moderate: `vector_size=100`, `epochs=8`, `min_count=3`.
"""
    )
)

cells.append(
    code(
        r"""EMB_KW = dict(
    vector_size=100,
    window=5,
    min_count=3,
    workers=4,
    epochs=8,
    seed=42,
)

# Skip-gram Word2Vec (same family as NB04 "semantic" model)
model_w2v = Word2Vec(sentences=sentences, sg=1, **EMB_KW)

# FastText: default subword range min_n=3, max_n=6; explicit for teaching clarity
model_ft = FastText(
    sentences=sentences,
    sg=1,
    min_n=3,
    max_n=6,
    bucket=2000000,
    **EMB_KW,
)

print("Word2Vec vocab size:", len(model_w2v.wv))
print("FastText  vocab size:", len(model_ft.wv))
print("FastText  vector_size:", model_ft.wv.vector_size)
"""
    )
)

cells.append(
    md(
        r"""# Part E — In‑vocabulary neighbors: **`mercy`**

If `mercy` is missing, inspect `model_w2v.wv.index_to_key[:30]` and pick another anchor, or lower `min_count`.
"""
    )
)

cells.append(
    code(
        r"""ANCHOR = "mercy"


def show_neighbors(title: str, wv, word: str, topn: int = 12):
    if word not in wv:
        print(f"{word!r} not in vocabulary — try another anchor or lower min_count.")
        return
    print(title)
    for w, sc in wv.most_similar(word, topn=topn):
        print(f"  {w:18s}  {sc:.4f}")
    print()


show_neighbors("Word2Vec (Skip-gram) neighbors", model_w2v.wv, ANCHOR)
show_neighbors("FastText (Skip-gram + subwords) neighbors", model_ft.wv, ANCHOR)
"""
    )
)

cells.append(
    md(
        r"""# Part F — OOV strings: Word2Vec fails lookup; FastText still returns a vector

We use a **synthetic string** that is extremely unlikely to appear as a whole word in the corpus. Word2Vec has **no row** → `KeyError`. FastText **composes** a vector from character n‑grams that *did* occur inside other trained words.

Then we ask FastText for **nearest neighbors** to that OOV vector. Treat the neighbor list as a **qualitative demo**, not ground truth.
"""
    )
)

cells.append(
    code(
        r"""OOV = "zzzmercylikezzz"

print("OOV demo token:", repr(OOV))
print()

try:
    _ = model_w2v.wv[OOV]
    print("Word2Vec: unexpectedly found a row (unexpected for this demo).")
except KeyError:
    print("Word2Vec: KeyError — no vector for this unseen word type.")

v_ft = model_ft.wv[OOV]
print("FastText: vector ok — shape", v_ft.shape, "L2 norm", float(np.linalg.norm(v_ft)))
print()

print("FastText most_similar to the OOV vector (qualitative):")
for w, sc in model_ft.wv.most_similar(positive=[OOV], topn=10):
    print(f"  {w:18s}  {sc:.4f}")
"""
    )
)

cells.append(
    md(
        r"""# Part G — Quick norm comparison (in‑vocab vs OOV)

FastText OOV vectors are still **real vectors**, but their scale and stability can differ from frequent in‑vocabulary words. A simple diagnostic is comparing **L2 norms** for a few tokens.
"""
    )
)

cells.append(
    code(
        r"""def l2_norm_wv(wv, w: str) -> float | None:
    if w not in wv:
        return None
    return float(np.linalg.norm(wv[w]))


rows = []
for label, wv in [("Word2Vec", model_w2v.wv), ("FastText", model_ft.wv)]:
    rows.append((label, "mercy", l2_norm_wv(wv, "mercy")))
    rows.append((label, OOV, l2_norm_wv(wv, OOV)))

tab = pd.DataFrame(rows, columns=["model", "token", "L2_norm"])
print(tab.to_string(index=False))

mercy_w2v = l2_norm_wv(model_w2v.wv, "mercy")
mercy_ft = l2_norm_wv(model_ft.wv, "mercy")
oov_ft = l2_norm_wv(model_ft.wv, OOV)

fig, ax = plt.subplots(figsize=(7, 3.8))
cats = ["mercy\n(in-vocab)", "OOV\n(synthetic)"]
x = np.arange(len(cats))
bw = 0.36
ax.bar(x - bw / 2, [mercy_w2v or 0.0, 0.0], bw, label="Word2Vec", color="#78909C", edgecolor="black")
ax.bar(x + bw / 2, [mercy_ft or 0.0, oov_ft or 0.0], bw, label="FastText", color="#26A69A", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylabel("L2 norm of word vector")
ax.set_title("Word2Vec has no row for OOV; FastText still returns a vector")
ax.legend()
ax.text(float(x[1] - bw / 2), 0.02, "KeyError\n(no lookup)", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md(
        r"""# Part H — Morphology ring: cosine similarity within a word family

We pick English surface forms around **`mercy`** that are **in the Word2Vec vocabulary** (so both models have a standard whole-word row). Then we compare **pairwise cosine similarity** matrices.

**How to read it:** higher off-diagonal similarity suggests the model places those forms in overlapping directions. FastText often tightens the ring because **shared character n‑grams** link the forms even before counting co-occurrence.
"""
    )
)

cells.append(
    code(
        r"""FAMILY = ["mercy", "merciful", "mercies", "mercifully"]

family_ok = [w for w in FAMILY if w in model_w2v.wv]
missing = [w for w in FAMILY if w not in model_w2v.wv]
if missing:
    print("Not in Word2Vec vocab (skipped for this matrix):", missing)
if len(family_ok) < 2:
    print("Need at least two family words in Word2Vec vocab — lower min_count or edit FAMILY.")
else:


    def cosim_matrix(wv, words: list[str]):
        M = np.stack([wv[w] for w in words], axis=0)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        return (M @ M.T, words)


    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (name, wv) in zip(
        axes,
        [("Word2Vec", model_w2v.wv), ("FastText", model_ft.wv)],
    ):
        C, labels = cosim_matrix(wv, family_ok)
        im = ax.imshow(C, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(f"{name}: cosine similarity\n(among family words)")
    fig.colorbar(im, ax=np.ravel(axes).tolist(), shrink=0.75, label="cosine")
    plt.tight_layout()
    plt.show()

    for name, wv in [("Word2Vec", model_w2v.wv), ("FastText", model_ft.wv)]:
        C, labels = cosim_matrix(wv, family_ok)
        dfm = pd.DataFrame(C, index=labels, columns=labels)
        print(name)
        print(dfm.round(3).to_string())
        print()
"""
    )
)

cells.append(
    md(
        r"""# Part I — Typo string: **`mercyy`**

Humans read typos; Word2Vec lookup is brittle. FastText can still **compose** a vector from character pieces that occurred inside real words during training.

Predict `most_similar` quality from **substring overlap**, not from “the model knows English spelling rules.”
"""
    )
)

cells.append(
    code(
        r"""TYPO = "mercyy"
print("Typo token:", repr(TYPO))
print()

try:
    _ = model_w2v.wv[TYPO]
    print("Word2Vec: vector exists (unexpected for this demo).")
except KeyError:
    print("Word2Vec: KeyError — no whole-word row for this typo.")

v_ty = model_ft.wv[TYPO]
print("FastText: composed vector — L2 norm", float(np.linalg.norm(v_ty)))
print("\nFastText neighbors of the typo vector:")
for w, sc in model_ft.wv.most_similar(positive=[TYPO], topn=10):
    print(f"  {w:18s}  {sc:.4f}")
"""
    )
)

cells.append(
    md(
        r"""# Part J — Sensitivity lab: `min_n` / `max_n` on a **small verse slice**

Retraining FastText on the **full** corpus with many hyperparameter grids is too slow for a one-hour block. Here we train **two small** FastText models on the **same first 3,500 verses** with fewer dimensions and epochs.

These models are **not** meant to match your full-corpus `model_ft` above. The goal is only to see **neighbor drift** when you change which character n‑grams are allowed.
"""
    )
)

cells.append(
    code(
        r"""SUB = sentences[: min(3500, len(sentences))]
LAB = dict(
    vector_size=48,
    window=5,
    min_count=3,
    workers=4,
    epochs=2,
    seed=101,
    sg=1,
    bucket=800000,
)

print("Subsample size (verses):", len(SUB))
print("Training shallow-subword FastText (min_n=2, max_n=4) ...")
ft_shallow = FastText(sentences=SUB, min_n=2, max_n=4, **LAB)
print("Training deep-subword FastText (min_n=5, max_n=8) ...")
ft_deep = FastText(sentences=SUB, min_n=5, max_n=8, **LAB)

PROBE = "mercy"
if PROBE not in ft_shallow.wv:
    PROBE = ft_shallow.wv.index_to_key[20]
    print("Probe not in shallow vocab; using:", repr(PROBE))

for title, m in [
    ("Shallow pieces: min_n=2, max_n=4", ft_shallow),
    ("Longer pieces: min_n=5, max_n=8", ft_deep),
]:
    print("=" * 56, title, "=" * 56)
    if PROBE not in m.wv:
        print("  probe missing in this model")
        continue
    for w, sc in m.wv.most_similar(PROBE, topn=8):
        print(f"  {w:18s}  {sc:.4f}")
    print()
"""
    )
)

cells.append(
    md(
        r"""# Part K — Mean‑pooled **verse** retrieval: Word2Vec vs FastText (ties to NB04)

Notebook 4 averaged **in‑vocabulary** Word2Vec vectors per verse. Here we implement the **same mean-pooling recipe** with two differences:

1. **Word2Vec:** skip unknown tokens (same as NB04).  
2. **FastText:** request a vector for **every** token; rare surface forms still get a composed vector.

Then we rank verses with **cosine similarity** for natural-language queries. Compare top hits and overlap.
"""
    )
)

cells.append(
    code(
        r"""def verse_vec_w2v_mean(toks: list[str], wv) -> np.ndarray:
    vecs = [wv[t] for t in toks if t in wv]
    if not vecs:
        return np.zeros(wv.vector_size)
    return np.mean(np.stack(vecs, axis=0), axis=0)


def verse_vec_ft_mean(toks: list[str], wv) -> np.ndarray:
    vecs = [wv[t] for t in toks]
    return np.mean(np.stack(vecs, axis=0), axis=0)


print("Building verse matrices (one row per training verse)...")
V_w2v = np.stack([verse_vec_w2v_mean(t, model_w2v.wv) for t in sentences])
V_ft = np.stack([verse_vec_ft_mean(t, model_ft.wv) for t in sentences])
print("Shapes:", V_w2v.shape, V_ft.shape)


def rank_verses(query: str, V: np.ndarray, vec_fn, wv, top_k: int = 5):
    qtoks = preprocess_text(query).split()
    qv = vec_fn(qtoks, wv)
    if float(np.linalg.norm(qv)) < 1e-9:
        return []
    sims = cosine_similarity(qv.reshape(1, -1), V)[0]
    top_idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i]), verse_ids[i], verse_texts[i][:100]) for i in top_idx]


def show_hits(title: str, hits: list) -> None:
    print(title)
    print("-" * 72)
    if not hits:
        print("(no hits — query vector all zeros?)")
        return
    for r, (i, sc, vid, snip) in enumerate(hits, 1):
        print(f"  {r}. [{sc:.4f}] {vid}  {snip}...")
    print()


for q in ("kindness", "divine compassion", "punishment of the wicked"):
    h2 = rank_verses(q, V_w2v, verse_vec_w2v_mean, model_w2v.wv)
    hf = rank_verses(q, V_ft, verse_vec_ft_mean, model_ft.wv)
    print("QUERY:", repr(q))
    show_hits("Word2Vec (skip unknown tokens in verse mean)", h2)
    show_hits("FastText (composed vector per token)", hf)
    id2 = {i for i, _, _, _ in h2}
    idf = {i for i, _, _, _ in hf}
    print("  Overlap of top-5 verse indices:", sorted(id2 & idf))
    print()
"""
    )
)

cells.append(
    md(
        r"""# Part L — PCA snapshot: same words, two geometries

We take **one shared word list** (frequent tokens in the corpus that exist in Word2Vec) and project each model’s vectors to **2D with PCA** separately.

**Do not over-interpret:** PCA is a linear projection for visualization; t‑SNE would bend neighborhoods differently. The point is to see that **subword training changes the global layout** of common words, not only rare ones.
"""
    )
)

cells.append(
    code(
        r"""flat = [t for s in sentences for t in s]
freq = Counter(flat)
HUB = [
    "mercy",
    "lord",
    "day",
    "night",
    "truth",
    "believers",
    "disbelievers",
    "fire",
    "garden",
    "paradise",
    "judgement",
    "fear",
    "hope",
]
words_plot: list[str] = []
for w, _ in freq.most_common(600):
    if w in model_w2v.wv and w not in words_plot:
        words_plot.append(w)
    if len(words_plot) >= 40:
        break
for h in HUB:
    if h in model_w2v.wv and h not in words_plot:
        words_plot.append(h)
words_plot = words_plot[:48]

mat_w2v = np.stack([model_w2v.wv[w] for w in words_plot])
mat_ft = np.stack([model_ft.wv[w] for w in words_plot])

pca_w = PCA(n_components=2, random_state=42).fit_transform(mat_w2v)
pca_f = PCA(n_components=2, random_state=42).fit_transform(mat_ft)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, xy, title in zip(
    axes,
    [pca_w, pca_f],
    ["Word2Vec (Skip-gram)", "FastText (Skip-gram + subwords)"],
):
    ax.scatter(xy[:, 0], xy[:, 1], alpha=0.45, s=28)
    for i, w in enumerate(words_plot):
        if i % 2 == 0 or w in set(HUB):
            ax.annotate(w, (xy[i, 0], xy[i, 1]), fontsize=6.5, alpha=0.85)
    ax.set_title(title + " — PCA 2D")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
plt.suptitle("Same word list, separate PCA fits", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md(
        r"""# Part M — Discussion prompts (≈10 minutes in groups)

1. When can **shared substrings mislead** FastText (false friends, unrelated words that look alike)?  
2. Why does **mean pooling** still fail to represent **negation** or **discourse** in a verse?  
3. If you **raised `min_count`**, which failure mode would worsen first: Word2Vec or FastText?  
4. Your slice models in Part J disagree with the full model—what does that teach about **data volume** vs **hyperparameters**?  
5. Where would you still want **sentence embeddings** (NB06) even if FastText retrieval looks good on short queries?

---
"""
    )
)

cells.append(
    code(
        r"""# --- Optional take-home exercises (uncomment / duplicate cells as needed) ---
# 1. Change FastText bucket (e.g. 500_000 vs 2_000_000) and compare OOV neighbor lists for the same synthetic token.
# 2. Pick three Arabic *transliteration* strings (same Latin script) and compare vectors / neighbors vs English query words.
# 3. Implement TF-IDF-weighted verse vectors (NB04 idea) and add a third retrieval column for one query.
# 4. Train FastText with sg=0 (CBOW) and compare mercy neighbors to sg=1 on the full corpus (longer run).
# 5. Log overlap@5 between W2V and FT retrieval for 10 queries; summarize when FT wins / loses.

print("Optional exercise prompts are in the comments above this cell.")
"""
    )
)

cells.append(
    md(
        r"""# Part N — What FastText still does **not** magically solve (bridge to sentence embeddings)

| Limitation | Notes |
|---|---|
| **Still static** | One representation per word type (plus subword composition); not contextual like a transformer layer. |
| **Sentence / discourse meaning** | Mean pooling over word vectors is a hack; negation, scope, and rhetoric are under-modeled. |
| **Hash collisions** | Large `bucket` reduces collisions but does not remove the approximation. |
| **Domain + bias** | Vectors reflect the training corpus statistics—including skew you may not want. |
| **Substring accidents** | Similar spelling does not imply similar meaning; subwords can reinforce spurious ties. |

```text
Notebook 5  →  FastText (subword + OOV-friendly vectors)
Notebook 6  →  Sentence embeddings (whole-text vectors; e.g. SBERT path in this series)
```

### Limitation → next step

| What you saw in NB05 | What is still missing | What comes next |
|---|---|---|
| Better rare / typo / OOV behavior at **word** level | Whole-query ↔ whole-verse meaning | **NB06 — sentence embeddings** |
| Mean-pooled verse retrieval | Learned **sentence encoders**, attention | **NB06+** |
| PCA geometry shifts | Efficient **nearest-neighbor search** at scale | **NB07 — FAISS** (in full series) |
| Single translation column | Grounded **generation** + citations | **NB08 — RAG** (in full series) |

---

## Mini self-check

- [ ] I can explain why Word2Vec raises **`KeyError`** on an unseen word type.  
- [ ] I can explain, in one paragraph, how FastText uses **character n‑grams**.  
- [ ] I can interpret **`most_similar`** for an OOV string as “shared substrings / statistics,” not as verified definitions.  
- [ ] I can read a **cosine matrix** for a word family and say one difference between W2V and FT.  
- [ ] I can explain why **verse mean pooling** behaves differently when unknown tokens appear in a verse.  
- [ ] I can name one case where **subwords mislead**.

When you are ready, open **Notebook 6 — Sentence Embeddings** (or the next notebook in your local track list).

---
"""
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

out_path = Path(__file__).parent / "NLP_Workshop_05_FastText_and_Subword_Embeddings.ipynb"
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote:", out_path)

if __name__ == "__main__":
    print("Done.")
