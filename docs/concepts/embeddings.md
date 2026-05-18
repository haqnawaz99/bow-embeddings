# What are Embeddings?

An **embedding** is a dense vector representation of a piece of text — a word, sentence, or document — that encodes its meaning as a point in a high-dimensional space. Things with similar meanings end up close together in this space.

---

## The Core Idea: Meaning as Position in Space

Imagine placing all the words in the English language on a map, where words with similar meanings are placed near each other:

```
                  compassion
          mercy ●   ● kindness
        ●
  forgiveness
                        ...far away...

                                        ● fire
                                    ● flame
                                  ● burn
```

An embedding is exactly this — a coordinate in a high-dimensional space. Instead of a 2D map, embeddings typically use 100 to 1,024 dimensions.

---

## The Distributional Hypothesis

The theoretical foundation for embeddings comes from linguistics:

> *"You shall know a word by the company it keeps."* — J.R. Firth, 1957

Words that appear in similar contexts tend to have similar meanings. "mercy" and "compassion" both appear near words like "forgive", "kind", "heart", "believers". This co-occurrence signal is what embedding models learn from.

---

## Three Levels of Embeddings

### 1. Word Embeddings (NB04, NB05)

One vector per word. Learned by training a model to predict words from context (Word2Vec) or using character n-grams (FastText).

```python
word2vec["mercy"]     # → 100-dimensional vector
word2vec["kindness"]  # → nearby vector
word2vec["elephant"]  # → distant vector
```

**Limitation:** One vector per word regardless of context. "bank" has the same vector whether it means a river bank or a financial institution.

### 2. Contextual Embeddings (BERT, etc.)

One vector per word *in context*. The same word gets different vectors depending on the surrounding sentence. "bank" next to "river" gets a different vector from "bank" next to "loan".

Not covered in this workshop (requires fine-tuning transformers), but this is what powers modern LLMs.

### 3. Sentence Embeddings (NB06)

One vector per sentence. Trained specifically to make semantically similar sentences map to nearby vectors.

```python
sbert.encode("Have mercy on the believers")
sbert.encode("Show compassion to those who believe")
# → these two sentences produce nearby vectors
```

This is the representation used for semantic search and RAG in NB07–NB09.

---

## Word2Vec: How Embeddings are Learned

Word2Vec learns embeddings by training a tiny neural network on a prediction task:

**CBOW (Continuous Bag of Words):** Given the surrounding words, predict the centre word.
```
Context: ["Allah", "is", ___, "merciful"]
Task: predict "most"
```

**Skip-Gram:** Given the centre word, predict the surrounding words.
```
Centre: "merciful"
Task: predict ["Allah", "is", "most", "and"]
```

The model never directly tries to learn what words mean. But to solve this prediction task well, it must learn that words used in similar contexts are similar. The network weights, after training, become the word embeddings.

---

## Analogy Arithmetic

A remarkable property of well-trained word embeddings: semantic relationships become vector arithmetic.

```
king - man + woman ≈ queen
Paris - France + Germany ≈ Berlin
mercy - forgiveness + punishment ≈ wrath
```

This works because the vector from "man" to "woman" encodes the gender relationship, and the same offset applies to "king" → "queen". Meaning is geometry.

---

## Sentence Embeddings: SBERT

Word embeddings cannot represent sentences — averaging word vectors loses word order, grammar, and composition.

SBERT (Sentence-BERT) trains a transformer to produce sentence-level embeddings by:
1. Processing the full sentence through a transformer
2. Mean-pooling the token representations
3. Fine-tuning on pairs of sentences labeled as similar/dissimilar

The result: sentences with similar meaning map to nearby vectors, even with no shared words.

```
"Patience in times of difficulty"
"Endurance through hardship"
→ cosine similarity ≈ 0.83
```

---

## Practical Properties

| Property | Detail |
|---|---|
| **Dimensionality** | Word2Vec: 100–300; SBERT: 384–768 |
| **L2 normalisation** | Normalize to unit length → dot product = cosine similarity |
| **Pre-trained models** | Download once, use everywhere. No training required. |
| **Language-specific** | Most models are English; multilingual models exist |
| **Domain shift** | General embeddings may perform poorly on domain-specific text |

---

→ Next: [What is RAG?](what-is-rag.md)
