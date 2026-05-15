# Claude Series — NLP Workshop Notebooks

This folder contains the primary workshop series. Run notebooks in order, starting from NB01.

| Notebook | Topic | Core techniques |
|---|---|---|
| [NB01](./NB01_Text_Preprocessing_and_BoW.ipynb) | Text Preprocessing and Bag-of-Words | Tokenization, stopwords, stemming, CountVectorizer |
| [NB02](./NB02_TF_IDF_and_Ranked_Retrieval.ipynb) | TF-IDF and Ranked Retrieval | TfidfVectorizer, IDF weighting, rank drift visualization |
| [NB03](./NB03_NGrams_and_Context.ipynb) | N-Grams and Context | Bigram features, language model, stopword trap |
| [NB04](./NB04_Word2Vec_Embeddings.ipynb) | Word2Vec Embeddings | CBOW vs Skip-Gram, PCA, t-SNE, analogy queries |
| [NB05](./NB05_FastText_and_Subword_Embeddings.ipynb) | FastText and Subword Embeddings | Character n-grams, OOV handling, L2 norm diagnostics |
| [NB06](./NB06_Sentence_Embeddings_and_Semantic_Search.ipynb) | Sentence Embeddings and Semantic Search | SBERT all-MiniLM-L6-v2, similarity histogram, TF-IDF vs SBERT overlap |
| [NB07](./NB07_Vector_Databases_and_FAISS.ipynb) | Vector Databases and FAISS | IndexFlatIP, IndexIVFFlat, nprobe sweep, recall@K, QuranSearchEngine |
| [NB08](./NB08_RAG_and_LLM_Retrieval.ipynb) | RAG and LLM-based Retrieval | Retrieve-Augment-Generate, flan-t5-small, RAG vs no-context comparison |
| [NB09](./NB09_Advanced_RAG_Techniques.ipynb) | Advanced RAG Techniques | Hybrid BM25+SBERT, RRF, cross-encoder re-ranking, HyDE, chunking, contextual compression, RAGAS-style evaluation |

## Prerequisites

Activate the virtual environment from the repo root before launching Jupyter:

```powershell
# from repo root
.\.venv\Scripts\Activate.ps1
jupyter notebook
```

## Dataset

All notebooks read `quran_translations.csv` from the parent folder. The path finder in each notebook handles relative paths automatically — no configuration needed.

## Visualizations

PNG files in this folder are generated outputs from the notebooks. They are committed for reference so you can compare your outputs to expected results without re-running everything.
