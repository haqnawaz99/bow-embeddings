# What is Natural Language Processing?

## Definition

**Natural Language Processing (NLP)** is a subfield of Artificial Intelligence concerned with the interaction between computers and human language. It involves developing algorithms and models that enable computers to:

- **Understand** the meaning and structure of text and speech
- **Generate** coherent and contextually appropriate language
- **Transform** language from one form to another (translation, summarization, etc.)

NLP sits at the intersection of **linguistics**, **computer science**, and **machine learning**.

---

## Real-World Applications

NLP powers technology you use every day:

| Application | Example |
|---|---|
| Search Engines | Understanding query intent, not just keywords |
| Machine Translation | Google Translate, DeepL |
| Virtual Assistants | Siri, Alexa, Google Assistant |
| Sentiment Analysis | Product review analysis at scale |
| Spam Detection | Gmail filtering |
| Question Answering | ChatGPT, Gemini, Claude |
| Named Entity Recognition | Extracting names/places from news |
| Text Summarization | Automatic news summarization |
| Autocomplete | Smartphone keyboard predictions |

---

## Why is NLP Hard?

Language appears simple to humans but is extraordinarily complex for machines.

### 1. Ambiguity

- *Lexical ambiguity*: "bank" can mean a river bank or a financial institution
- *Syntactic ambiguity*: "I saw the man with a telescope" — who has the telescope?
- *Semantic ambiguity*: "The chicken is ready to eat" — is the chicken hungry, or is it cooked?

### 2. Context Dependence

- "It's cold" means very different things in a weather report vs. a description of someone's personality
- Pronouns and coreference: "The trophy didn't fit in the suitcase because it was too big" — what does "it" refer to?

### 3. Language Variation

- Dialects, slang, informal writing, abbreviations
- Sarcasm and irony: "Oh great, another Monday"
- Domain-specific language: legal text, medical jargon, poetry

### 4. World Knowledge

- Understanding "The Eiffel Tower is in Paris" requires knowing geography
- Implicit references and cultural context

### 5. Morphological Complexity

- "run", "runs", "running", "ran" all refer to the same concept
- In Arabic, a single word can encode what English expresses in a full sentence

---

## A Brief History of NLP

### The Rule-Based Era (1950s–1980s)
Early NLP systems were entirely hand-crafted. Linguists wrote explicit grammar rules and vocabulary lists. Systems like ELIZA (1966) mimicked conversation through pattern matching but had no real language understanding.

### The Statistical Revolution (1990s)
The 1990s saw a dramatic shift. Researchers discovered that statistical models trained on large corpora could outperform hand-crafted rules:

- **1996**: Bag-of-Words becomes standard for document classification
- **1998**: TF-IDF weighting becomes widespread in Information Retrieval
- **1999**: The vector space model dominates search engines

### The Machine Learning Era (2000s)
- **2001**: SVMs applied to text classification
- **2003**: Latent Dirichlet Allocation (LDA) for topic modeling
- **2008**: Sentiment analysis becomes a major subfield

### The Deep Learning Era (2013–2017)
- **2013**: Word2Vec — dense semantic word vectors
- **2014**: GloVe — global vector representations
- **2015**: Attention mechanisms emerge
- **2017**: FastText — subword embeddings handle OOV

### The Transformer Era (2018–present)
- **2018**: BERT — bidirectional transformers, contextual embeddings
- **2019**: SBERT — sentence-level embeddings for fast semantic search
- **2020**: GPT-3 — large language models at scale
- **2022**: ChatGPT — conversational LLMs go mainstream
- **2023**: RAG becomes the standard for grounding LLMs in real data

---

## The Workshop Arc

This workshop teaches NLP in historical order — each technique motivated by the failure of the previous one. By the time you reach RAG in NB09, you will understand not just *how* it works but *why* every design decision was made.

→ Start with [NB01: Text Preprocessing and Bag-of-Words](../notebooks/nb01.md)
