# Invenza

Invenza is a local-first patent retrieval and business-idea generation system. It searches a PostgreSQL database with the `pgvector` extension, uses Ollama to generate patent-grounded ideas, and calculates an explainable grounding score from retrieval and text-alignment signals.

The repository currently contains the backend pipeline, ingestion/search utilities, training-data preparation scripts, embedding fine-tuning artifacts, and QLoRA fine-tuning support. A frontend application is planned but is not implemented yet.

## What Is Implemented

- CSV patent ingestion into PostgreSQL with `pgvector`
- HNSW vector index creation for semantic search
- Sentence-transformers embeddings for patent and query retrieval
- Patent-grounded idea generation through Ollama
- Query-complexity model routing:
  - short queries use `qwen3:4b`
  - longer queries use `llama3.2`
  - failed requests fall back to the other model
- Hybrid grounding score from 0 to 100 using:
  - retrieval similarity: 45%
  - citation overlap: 25%
  - topic alignment: 20%
  - text consistency: 10%
- FastAPI endpoints for health checks and idea generation
- Optional embedding fine-tuning and QLoRA training workflows
- Router and multi-model integration tests

## Architecture

```mermaid
flowchart LR
    CSV[Patent CSV files] --> INGEST[Ingestion script]
    INGEST --> DB[(PostgreSQL + pgvector)]
    QUERY[Topic or idea request] --> EMBED[Sentence-transformers]
    EMBED --> RETRIEVE[Vector retrieval]
    DB --> RETRIEVE
    RETRIEVE --> ROUTER[Model router]
    ROUTER --> OLLAMA[Ollama LLM]
    OLLAMA --> SCORE[Hybrid grounding scorer]
    RETRIEVE --> SCORE
    SCORE --> API[FastAPI response]
```

## Repository Layout

```text
.
|-- gp-search-*.csv                 Patent export files
|-- README.md                       Project documentation
|-- IMPLEMENTATION_SUMMARY.md       Implementation notes and quick-start history
|-- Backend/
|   |-- app/
|   |   |-- api.py                  FastAPI application
|   |   |-- config.py               Environment-backed settings
|   |   |-- grounding.py            Grounding score calculation
|   |   |-- ingestion.py             Database/table helpers
|   |   |-- llm.py                   Ollama and local Qwen clients
|   |   |-- model_router.py          Query routing and fallback logic
|   |   |-- pipeline.py              End-to-end idea generation
|   |   |-- retrieval.py             pgvector semantic search
|   |   `-- training.py              Training helpers
|   |-- ingest_patents.py            Load CSV files into PostgreSQL
|   |-- search_patents.py            Command-line semantic search
|   |-- scripts/                     Data preparation and fine-tuning scripts
|   |-- tests/                       Backend tests
|   |-- data/                        Generated/training datasets
|   |-- models/                      Local model artifacts
|   |-- config/                      Fine-tuning configuration
|   |-- docker-compose.yml           PostgreSQL and API containers
|   |-- Dockerfile
|   |-- requirements.txt
|   |-- README.md                    Backend-specific ingestion notes
|   `-- FINETUNING_GUIDE.md          Detailed fine-tuning instructions
`-- Frontend/                       Reserved for the future UI; currently empty
```

## Requirements

### Basic local run

- Windows, macOS, or Linux
- Python 3.10 or newer
- PostgreSQL 16 with the `pgvector` extension, or the included Docker image
- Ollama
- An Ollama model, at minimum `llama3.2`
- Enough disk space for the embedding model and patent database

### Fine-tuning

- NVIDIA GPU recommended
- At least 8 GB VRAM recommended for QLoRA
- Additional storage for datasets and model artifacts

## Installation

From PowerShell:

```powershell
Set-Location Backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install and start Ollama separately, then pull the default model:

```powershell
ollama pull llama3.2
```

The router can select `qwen3:4b` for simple requests. Pull it when using the router with both Ollama models:

```powershell
ollama pull qwen3:4b
```

## Configuration

Create `Backend/.env`:

```dotenv
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_BASE_URL=http://localhost:11434
CSV_DIR=..
BATCH_SIZE=128
DEFAULT_TOPIC=precision agriculture
```

Important settings:

| Variable | Purpose | Default |
| --- | --- | --- |
| `DATABASE_URL` | PostgreSQL connection string | Docker database URL |
| `EMBEDDING_MODEL` | Hugging Face model name or local model path | `sentence-transformers/all-MiniLM-L6-v2` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://host.docker.internal:11434` |
| `CSV_DIR` | Directory containing `gp-search-*.csv` files | repository root |
| `BATCH_SIZE` | Ingestion batch size | `128` |
| `DEFAULT_TOPIC` | Topic used by the CLI when no topic is supplied | `precision agriculture` |

## Database and Patent Ingestion

The database must have `pgvector` enabled:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

With PostgreSQL running, ingest all patent CSV files from the repository root:

```powershell
Set-Location Backend
python ingest_patents.py
```

Ingest one file or choose another directory:

```powershell
python ingest_patents.py --csv ..\gp-search-Artificial-intelligence.csv
python ingest_patents.py --csv-dir ..
```

The loader searches for `gp-search-*.csv`, skips the leading search URL row, creates the `patents` table and HNSW vector index, and upserts records in batches.

## Semantic Search

Search the database directly from the command line:

```powershell
Set-Location Backend
python search_patents.py "image recognition artificial intelligence"
python search_patents.py "precision agriculture sensors" --limit 10
```

Results include the patent ID, title, assignee, filing date, cosine similarity, and L2 distance.

## Run the API

From `Backend`:

```powershell
python -m uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Check that the service is running:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Generate a patent-grounded idea:

```powershell
$body = @{ topic = "IoT sensors for sustainable agriculture"; top_k = 2 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ideas -ContentType "application/json" -Body $body
```

The response contains:

- `concept_name` and `summary`
- `source_patents`
- `grounding_score`
- `passes_target`
- `evidence_summary`
- `retrieval_details`
- `model_used`
- `inference_time_ms`

Interactive API documentation is available at `http://localhost:8000/docs`.

## Run the Pipeline from the Command Line

```powershell
Set-Location Backend
python -m app.pipeline --topic "precision agriculture"
```

Useful options:

```powershell
python -m app.pipeline --topic "medical imaging"
python -m app.pipeline --topic "smart farming" --force-model llama3.2
python -m app.pipeline --topic "smart farming" --enable-qwen
```

## Model Routing

The router currently uses a 50-token threshold:

- queries under the threshold route to `qwen3:4b`
- queries at or above the threshold route to `llama3.2`
- if the selected model fails, the pipeline retries with the fallback model

The local Qwen path is optional. The default pipeline uses Ollama and does not load a local Qwen model unless `--enable-qwen` is passed.

## Fine-Tuning Workflow

The fine-tuning workflow is optional and requires additional compute. The intended order is:

```powershell
Set-Location Backend
python scripts/extract_training_data.py
python scripts/prepare_embedding_dataset.py
python scripts/finetune_embeddings.py
python scripts/re_embed_patents.py
python scripts/finetune_qwen.py
```

For a smaller smoke test:

```powershell
python scripts/finetune_qwen.py --max-samples 1000
```

The workflow supports:

1. Extracting patent records from PostgreSQL into JSONL
2. Building embedding triplets
3. Fine-tuning a sentence-transformers embedding model
4. Re-embedding stored patents
5. Fine-tuning Qwen with QLoRA
6. Routing simple and complex queries to different models

See [Backend/FINETUNING_GUIDE.md](Backend/FINETUNING_GUIDE.md) for GPU requirements, data formats, hyperparameters, and troubleshooting.

## Testing

Run the router and integration tests:

```powershell
Set-Location Backend
python -m pytest tests/test_router.py -v
```

Run all available backend tests:

```powershell
python -m pytest tests -v
```

## Docker

The backend compose file provides PostgreSQL with pgvector and the API service:

```powershell
Set-Location Backend
docker compose up --build
```

The API is then available at `http://localhost:8000` and PostgreSQL at `localhost:5432`.

The compose file also contains a frontend service definition, but the `Frontend` directory is currently empty. The frontend service will need a Dockerfile and application before it can be started successfully.

## Current Limitations

- The API request includes `top_k`, but the current pipeline uses a fixed retrieval limit of 2.
- Grounding is currently formula-based and explainable; a trained transformer grounding scorer is part of the planned evolution, not the active inference path.
- The generated prompt currently sends retrieved titles as the evidence text. Full abstracts and claims are not yet passed through the idea-generation prompt.
- The API creates a new pipeline instance for each `/ideas` request, so model and embedding loading may be repeated.
- A working frontend has not yet been added.
- Exact latency and quality depend on the local hardware, Ollama models, database size, and embedding model.

## Next Development Steps

- Add the frontend application and connect it to `/ideas`.
- Honor `top_k` throughout the API and pipeline.
- Cache embedding models and pipeline clients between requests.
- Pass full patent abstracts/claims into retrieval evidence where available.
- Add persisted grounding-model training and evaluation.
- Add API tests for `/health` and `/ideas` with mocked database and LLM clients.

## Related Documentation

- [Backend README](Backend/README.md)
- [Implementation summary](IMPLEMENTATION_SUMMARY.md)
- [Fine-tuning guide](Backend/FINETUNING_GUIDE.md)
- [QLoRA configuration](Backend/config/qwen_qlora_config.json)
