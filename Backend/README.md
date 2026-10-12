# Patent CSV to PostgreSQL + pgvector

## Setup

1. Enable the extension in the target database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

2. Install the pipeline dependencies:

```powershell
cd Backend
python -m pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set `DATABASE_URL` to the database shown in pgAdmin.

## Ingest all CSV files

From the `Backend` directory:

```powershell
python ingest_patents.py
```

The loader searches `CSV_DIR` for `gp-search-*.csv`, skips the leading search URL row, creates the `patents` table and HNSW vector index, and upserts records in batches. The first embedding-model run downloads the model from Hugging Face and caches it locally.

To ingest one file or override the CSV directory:

```powershell
python ingest_patents.py --csv ..\gp-search-Artificial-intelligence.csv
python ingest_patents.py --csv-dir ..
```

## Semantic search

```powershell
python search_patents.py "image recognition artificial intelligence"
```

The search script embeds the query and uses pgvector cosine distance (`<=>`) to return the nearest patents.
