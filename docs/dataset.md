# Dataset

## Sample dataset

The workshop ships with a ready-to-use CSV file so you can run every notebook immediately without finding your own data.

```
quran_translations.csv  (included in the repo root)
Rows: 6,236
```

| Column | Description |
|---|---|
| `Surah` | Chapter number |
| `Verse` | Verse number within chapter |
| `ahmedali`, `arberry`, `daryabadi` ... | 16 different English translations |

All notebooks default to one translation column. You can change it in the config cell at the top of any notebook:

```python
TEXT_COLUMN = "daryabadi"   # change to any column name from the CSV
```

Available translation columns: `ahmedali`, `ahmedraza`, `arberry`, `daryabadi`, `hilali`, `itani`, `maududi`, `mubarakpuri`, `pickthall`, `qarai`, `qaribullah`, `sahih`, `sarwar`, `shakir`, `wahiduddin`, `yusufali`

---

## Bring your own data

The entire pipeline is dataset-agnostic. To swap in your own corpus, replace the CSV loading cell in any notebook:

```python
# Original
df = pd.read_csv(CSV_PATH)[["Surah", "Verse", "daryabadi"]].dropna()
df["verse_id"] = df["Surah"].astype(str) + ":" + df["Verse"].astype(str)

# Your data — two columns minimum: an ID and a text column
df = pd.read_csv("your_data.csv")[["id", "text"]].dropna()
df["verse_id"] = df["id"].astype(str)   # used for citations
```

Everything else in the notebook — preprocessing, indexing, retrieval, generation — works unchanged.

---

## What makes a good corpus for this workshop

| Property | Why it matters |
|---|---|
| Short to medium passages (50–500 words each) | Retrieval works best at paragraph or sentence level |
| At least 1,000 rows | Enough variety to make retrieval non-trivial |
| Plain English text | Preprocessing assumes English stopwords and tokenisation |
| A meaningful ID column | Used for citations in RAG prompts |

---

## Example corpora you could use

- **News articles** — one row per article or paragraph
- **Legal documents** — one row per clause or section
- **Research paper abstracts** — one row per abstract
- **Product reviews** — one row per review
- **Wikipedia passages** — one row per section
- **Company knowledge base** — one row per FAQ entry or policy section

The more semantically rich your corpus, the more interesting the semantic search and RAG results will be.
