# Sparse vs Dense Representations

One of the most fundamental transitions in NLP history was the shift from **sparse** to **dense** vector representations. Understanding this distinction is the key to understanding why modern NLP works the way it does.

---

## Sparse Representations (BoW, TF-IDF, N-Grams)

In a sparse representation, each word in the vocabulary gets its own dimension. A document is represented as a vector where most values are zero.

```
Vocabulary: [allah, mercy, kind, fire, water, camel, ...]
                                       (5,000 words total)

A verse about mercy:
[0.8, 0.6, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]
  ↑    ↑    ↑
 few non-zero values          ~4,997 zeros
```

**Properties:**
- Dimensionality = vocabulary size (typically 5,000–100,000)
- Most values are zero (99%+ zeros for typical documents)
- Each dimension has a clear, interpretable meaning ("this is the 'mercy' dimension")
- No notion of similarity between dimensions — "mercy" and "kindness" are orthogonal

---

## Dense Representations (Word2Vec, FastText, SBERT)

In a dense representation, every word or sentence is mapped to a low-dimensional vector where **every dimension has a value**. No zeros.

```
Word2Vec representation of "mercy":
[0.23, -0.41, 0.87, 0.12, -0.65, 0.34, ..., 0.09]
  ↑     ↑      ↑     ↑     ↑      ↑           ↑
               ALL 100 dimensions have values
```

**Properties:**
- Dimensionality = 50–1,024 (typically 100–384 for most tasks)
- Every dimension is non-zero
- Dimensions are not directly interpretable (no single dimension = "mercy")
- Similar meanings → similar vectors (nearby in vector space)

---

## The Key Difference: Semantic Similarity

This is where dense representations win decisively.

In sparse space:

```
"mercy"       → [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ...]
"kindness"    → [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, ...]
"elephant"    → [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, ...]

cosine("mercy", "kindness") = 0   ← no shared dimensions
cosine("mercy", "elephant") = 0   ← same score!
```

In dense space (Word2Vec trained on text):

```
cosine("mercy", "kindness")    = 0.76   ← high similarity
cosine("mercy", "compassion")  = 0.81   ← even higher
cosine("mercy", "elephant")    = 0.08   ← low similarity
```

---

## Why Sparse Representations Were Used First

Sparse representations have real advantages:

| Advantage | Detail |
|---|---|
| **Interpretable** | You know exactly which words contributed to a score |
| **No training needed** | Count words — no neural network required |
| **Exact keyword matching** | Perfect recall for exact terms |
| **Fast to build** | Minutes on a laptop, even for large corpora |
| **Works on any language** | No pre-trained model needed |

TF-IDF still outperforms dense embeddings for **exact keyword queries**. This is why production systems often use both (hybrid retrieval — covered in NB09).

---

## The Curse of Dimensionality

One major problem with sparse high-dimensional vectors: distances become less meaningful as dimensions increase.

In 2D, "close" and "far" are intuitive. In 5,000 dimensions, all vectors tend to be roughly the same distance from each other. This makes nearest-neighbour search (finding the most similar document) unreliable for very high-dimensional sparse vectors.

Dense representations at 100–384 dimensions avoid this problem.

---

## Transition in This Workshop

| Notebook | Representation | Type |
|---|---|---|
| NB01 | Bag-of-Words counts | Sparse |
| NB02 | TF-IDF weights | Sparse |
| NB03 | N-gram counts | Sparse |
| NB04 | Word2Vec 100-dim | **Dense** |
| NB05 | FastText 100-dim | **Dense** |
| NB06 | SBERT 384-dim | **Dense** |
| NB07–09 | SBERT + FAISS | **Dense** |

The transition from NB03 to NB04 is the biggest conceptual leap in the series.

---

→ Next: [What are Embeddings?](embeddings.md)
