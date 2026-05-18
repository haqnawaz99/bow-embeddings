# Setup

## Requirements

| Requirement | Detail |
|---|---|
| Python | 3.10 or higher |
| GPU | Not required — all notebooks run on CPU |
| API key | Not required — local models only |
| Disk space | ~1 GB for models + dependencies |
| RAM | 4 GB minimum, 8 GB recommended |

---

## Installation

=== "Windows (PowerShell)"

    ```powershell
    git clone https://github.com/haqnawaz99/BOW-to-Embeddings.git
    cd BOW-to-Embeddings

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

    pip install -r requirements.txt
    ```

    !!! warning "PyTorch on Windows"
        The default PyTorch from pip may fail on some Windows machines with a DLL error. Use the CPU-only build instead:

        ```powershell
        pip install torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu `
            --index-url https://download.pytorch.org/whl/cpu
        ```

=== "macOS / Linux"

    ```bash
    git clone https://github.com/haqnawaz99/BOW-to-Embeddings.git
    cd BOW-to-Embeddings

    python3 -m venv .venv
    source .venv/bin/activate

    pip install -r requirements.txt
    ```

---

## Launch Jupyter

```bash
jupyter notebook
```

Navigate to the `Claude/` folder and open `NB01_Text_Preprocessing_and_BoW.ipynb`.

In VS Code, select the interpreter at `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (macOS/Linux).

---

## First-run model downloads

Some notebooks download models automatically on first use. This only happens once — models are cached locally by HuggingFace.

| Model | Size | Used in |
|---|---|---|
| NLTK punkt tokenizer | ~2 MB | NB01–NB09 |
| `all-MiniLM-L6-v2` sentence encoder | ~90 MB | NB06–NB09 |
| `google/flan-t5-small` generator | ~300 MB | NB08–NB09 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~85 MB | NB09 |

Total first-run download: approximately 500 MB.

---

## Verify your setup

Run this in a terminal with your virtual environment active:

```python
python -c "
import nltk, sklearn, gensim, sentence_transformers, faiss, transformers, rank_bm25
print('All dependencies OK')
"
```

If all imports succeed without errors, you are ready to run the notebooks.

---

## Troubleshooting

**`ModuleNotFoundError` for any package**
```bash
pip install -r requirements.txt --upgrade
```

**NLTK data not found**
Each notebook auto-downloads required NLTK data. If this fails due to network restrictions:
```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

**FAISS import error on Windows**
Use `faiss-cpu` (already in `requirements.txt`). Do not install `faiss-gpu` unless you have a CUDA GPU configured.

**flan-t5-small download fails**
Set `USE_LOCAL_LLM = False` in the NB08/NB09 config cell. The retrieval sections will still run — only the generation cells will be skipped.
