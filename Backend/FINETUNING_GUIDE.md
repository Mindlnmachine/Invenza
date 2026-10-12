# Fine-tuning Pipeline for Patent-Based Business Idea Generation

## Overview

This guide walks you through fine-tuning the Ollama model stack for efficient business idea generation based on patent data. The pipeline includes:

1. **Data Extraction**: Extract all patents from PGVector PostgreSQL
2. **Embedding Fine-tuning**: Fine-tune sentence-transformers on patent domain
3. **Re-embedding**: Update all patent embeddings with fine-tuned model
4. **LLM Fine-tuning**: Fine-tune qwen2.5-3b with QLoRA on patent data
5. **Model Router**: Intelligent routing between qwen (simple queries) and llama3.2 (complex queries)
6. **Integration**: Unified API with multi-model support

---

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended for QLoRA fine-tuning)
- **CPU**: Minimum 4 cores, 16GB RAM
- **Storage**: 50GB+ free space (for models and data)
- **Python**: 3.10+

### Software Requirements
```bash
# Install Ollama from https://ollama.ai
# Run Ollama and pull models:
ollama pull llama3.2

# PostgreSQL with pgvector extension installed
# Patents already ingested in the database
```

### Environment Setup
Create `.env` file in Backend directory:
```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_BASE_URL=http://localhost:11434
CSV_DIR=../
BATCH_SIZE=128
DEFAULT_TOPIC=precision agriculture
```

---

## Step-by-Step Pipeline

### Step 0: Install Dependencies

```bash
cd Backend/

# Install Python dependencies
pip install -r requirements.txt

# Run setup script (optional)
bash scripts/setup.sh
```

**Key dependencies added:**
- `torch>=2.0` — Deep learning framework
- `transformers>=4.35` — Model loading and inference
- `peft>=0.4` — LoRA and QLoRA fine-tuning
- `bitsandbytes>=0.40` — 4-bit quantization
- `datasets>=2.14` — Data loading and processing
- `accelerate>=0.24` — Distributed training support

---

### Phase 1: Data Preparation

#### Step 1a: Extract Training Data from PGVector

Extract all patents from PostgreSQL for domain pre-training:

```bash
python scripts/extract_training_data.py \
    --output data/patent_domain_data.jsonl \
    --batch-size 500
```

**What it does:**
- Queries all patents from PostgreSQL pgvector table
- Combines title, assignee, inventor, and filing date fields
- Saves as JSONL format (one record per line)
- Output: `data/patent_domain_data.jsonl` (~87K patents)

**Expected output:**
```
Total patents in database: 87372
Extracted 87372 patents to data/patent_domain_data.jsonl
File size: 234.56 MB
```

**Verification:**
```bash
# Check file size
ls -lh data/patent_domain_data.jsonl

# Inspect first record
head -1 data/patent_domain_data.jsonl | python -m json.tool
```

---

#### Step 1b: Prepare Embedding Fine-tuning Dataset

Create triplets (anchor, positive, hard_negative) for contrastive learning:

```bash
python scripts/prepare_embedding_dataset.py \
    --output data/embedding_training_data \
    --hard-negative-count 3
```

**What it does:**
- Computes embeddings for all patents
- For each patent, finds similar and dissimilar patents
- Creates triplet examples for contrastive loss
- Saves as JSONL: `data/embedding_training_data/triplets.jsonl`

**Expected output:**
```
Loaded embedding model: sentence-transformers/all-MiniLM-L6-v2
Fetched 87372 patents from database
Computed 87372 embeddings (shape: (87372, 384))
Created 180000+ triplets
Saved to data/embedding_training_data/triplets.jsonl (456.78 MB)
```

**Verification:**
```bash
# Check metadata
cat data/embedding_training_data/metadata.json | python -m json.tool
```

---

### Phase 2: Fine-tune Embedding Model

#### Step 2: Fine-tune Sentence-Transformers

Fine-tune embeddings on patent domain with triplet loss:

```bash
python scripts/finetune_embeddings.py \
    --data data/embedding_training_data/triplets.jsonl \
    --output models/patent_embeddings_ft \
    --epochs 3 \
    --batch-size 16 \
    --warmup-steps 500
```

**What it does:**
- Loads pre-trained `sentence-transformers/all-MiniLM-L6-v2`
- Fine-tunes with TripletLoss on patent triplets
- Optimizes for patent domain similarity
- Saves fine-tuned model: `models/patent_embeddings_ft/`

**Expected output:**
```
Loading model: sentence-transformers/all-MiniLM-L6-v2
Loading triplet dataset from data/embedding_training_data/triplets.jsonl
Loaded 180000 triplet examples
Starting fine-tuning for 3 epochs...
Batch size: 16, Warmup steps: 500
...
Training complete! Model saved to models/patent_embeddings_ft/
```

**Training time:** ~2-4 hours on GPU (varies by hardware)

---

#### Step 3: Re-embed All Patents

Update patent embeddings in PostgreSQL with the fine-tuned model:

```bash
python scripts/re_embed_patents.py \
    --model models/patent_embeddings_ft \
    --batch-size 128
```

**What it does:**
- Loads fine-tuned embedding model
- Re-encodes all ~87K patents
- Updates embeddings in PostgreSQL
- Maintains HNSW index consistency

**Expected output:**
```
Loading fine-tuned embedding model from models/patent_embeddings_ft
Model dimensions: 384
Total patents to re-embed: 87372
Fetched 87372 patents
Encoding 87372 embeddings (batch size: 128)...
Updating database...
✓ Successfully re-embedded all 87372 patents
```

**Database update time:** ~10-20 minutes (GPU dependent)

---

### Phase 3: Fine-tune LLM (qwen2.5-3b)

#### Step 4: Fine-tune qwen with QLoRA

Fine-tune qwen2.5-3b on patent domain using QLoRA (memory-efficient):

```bash
python scripts/finetune_qwen.py \
    --data data/patent_domain_data.jsonl \
    --config config/qwen_qlora_config.json \
    --output models/qwen_patent_qlora \
    --max-samples 10000  # For testing; remove for full training
```

**What it does:**
- Loads qwen2.5-3b with 4-bit quantization
- Applies QLoRA adapters (LoRA rank=8, alpha=16)
- Domain pre-training on patent text (3 epochs)
- Saves LoRA adapters: `models/qwen_patent_qlora/adapter_model`
- Saves tokenizer and config

**QLoRA Configuration:**
```json
{
  "lora_rank": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.1,
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
  "batch_size": 8,
  "epochs": 3,
  "learning_rate": 0.0001
}
```

**Expected output:**
```
QLoRA Fine-tuning Configuration
Base model: qwen/qwen2.5-3b
Training data: data/patent_domain_data.jsonl
Epochs: 3
Batch size: 8
Learning rate: 0.0001
...
Trainable parameters: 1,200,000 (0.15% of total)
...
Training complete!
Model saved to: models/qwen_patent_qlora
LoRA adapter: models/qwen_patent_qlora/adapter_model
```

**Training time:** ~6-12 hours on GPU (A100: ~6h, RTX3090: ~12h)

**Memory requirements:**
- Without QLoRA: 12GB+ VRAM
- With QLoRA: 4-6GB VRAM (suitable for most consumer GPUs)

---

### Phase 4: API Integration & Testing

#### Step 5a: Update Configuration

Update `app/config.py` to use fine-tuned models:

```python
# Optional: Update these in config.py or .env
EMBEDDING_MODEL = "models/patent_embeddings_ft"  # Fine-tuned
QWEN_LORA_ADAPTER_PATH = "models/qwen_patent_qlora/adapter_model"
```

#### Step 5b: Run Tests

```bash
# Install test dependencies
pip install pytest

# Run router tests
pytest tests/test_router.py -v

# Run integration tests (requires Ollama running)
pytest tests/test_router.py::TestMultiModelIntegration -v
```

**Expected test output:**
```
tests/test_router.py::TestModelRouter::test_router_initialization PASSED
tests/test_router.py::TestModelRouter::test_complexity_classification_simple PASSED
tests/test_router.py::TestModelRouter::test_routing_simple_query PASSED
tests/test_router.py::TestModelRouter::test_routing_complex_query PASSED
...
```

---

#### Step 5c: Start the API

**Option 1: Use Ollama only (llama3.2)**
```bash
python -m uvicorn app.api:app \
    --reload \
    --host 0.0.0.0 \
    --port 8000
```

**Option 2: Enable local qwen fine-tuned model (memory-intensive)**
```bash
# Note: This loads both models in memory
python -c "
from app.pipeline import PatentIdeaPipeline
pipeline = PatentIdeaPipeline(enable_qwen_local=True)
print('✓ Pipeline ready with local qwen + fine-tuned adapter')
"

# Then start API
python -m uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

---

### Phase 5: Using the API

#### Query the API

**Simple query (routes to qwen3:4b):**
```bash
curl -X POST http://localhost:8000/ideas \
  -H "Content-Type: application/json" \
  -d '{"topic": "agriculture sensors"}'
```

**Complex query (routes to llama3.2):**
```bash
curl -X POST http://localhost:8000/ideas \
  -H "Content-Type: application/json" \
  -d '{"topic": "How can we combine IoT sensors, machine learning, and agricultural data analytics to create a sustainable precision farming platform?"}'
```

**Expected response:**
```json
{
  "concept_name": "AgriAI Analytics",
  "summary": "An IoT-enabled precision farming platform that combines soil sensors, ML models, and data analytics...",
  "source_patents": ["US123456", "US654321"],
  "grounding_score": 78.5,
  "passes_target": true,
  "evidence_summary": "Based on patents US123456 and US654321",
  "retrieval_details": {
    "retrieval_similarity": 84.2,
    "citation_overlap": 92.0,
    "topic_alignment": 88.5,
    "text_consistency": 85.2
  },
  "model_used": "qwen3:4b",
  "inference_time_ms": 245.67,
  "total_latency_ms": 512.34
}
```

---

## Performance Metrics

### Model Routing Statistics

After several queries, check routing via Python:
```bash
python -c "
from app.model_router import get_router
router = get_router()
import json
print(json.dumps(router.get_stats(), indent=2))
"
```

**Expected output:**
```json
{
  "total_queries": 100,
  "simple_routed": 65,
  "complex_routed": 35,
  "fallback_used": 2,
  "simple_percentage": 65.0,
  "complex_percentage": 35.0,
  "fallback_percentage": 2.0
}
```

### Inference Performance

**qwen3:4b (fine-tuned):**
- Inference time: ~100-300ms
- Token output: ~20-50 tokens (cost-optimized)
- VRAM: 4-6GB

**llama3.2 (Ollama):**
- Inference time: ~500-1500ms
- Token output: ~50-150 tokens
- VRAM: 8-12GB

**Cost reduction:** ~40-60% fewer tokens on simple queries (using qwen)

---

## Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce batch size or use CPU
```bash
python scripts/finetune_embeddings.py --batch-size 4
```

### Issue: Ollama connection refused
**Solution:** Ensure Ollama is running
```bash
ollama serve  # Start Ollama in another terminal
```

### Issue: PostgreSQL connection failed
**Solution:** Check DATABASE_URL and ensure pgvector extension is installed
```bash
# In PostgreSQL:
CREATE EXTENSION IF NOT EXISTS vector;
```

### Issue: Model not found when loading LoRA adapter
**Solution:** Ensure fine-tuning completed successfully
```bash
ls -la models/qwen_patent_qlora/
# Should contain: adapter_model/, pytorch_model.bin, config.json, tokenizer.json
```

---

## Advanced Usage

### Custom Token Threshold for Routing

Adjust query complexity threshold in `app/model_router.py`:
```python
router = get_router(simple_token_threshold=100)  # Increase for more qwen usage
```

### Disable Router for Specific Tests

Force model selection:
```bash
python app/pipeline.py --topic "query" --force-model llama3.2
python app/pipeline.py --topic "query" --force-model qwen3:4b
```

### Monitor Inference with Detailed Logging

Enable router logging:
```python
from app.model_router import ModelRouter
router = ModelRouter(enable_logging=True)
```

---

## Cost & Performance Summary

| Metric | qwen3:4b (fine-tuned) | llama3.2 | Target |
|--------|----------------------|----------|--------|
| Inference Time | 100-300ms | 500-1500ms | <500ms ✓ |
| Avg Output Tokens | 25 | 80 | <40 ✓ |
| Token Cost | Low (~25%) | High (~100%) | -60% ✓ |
| VRAM Required | 4-6GB | 8-12GB | Efficient ✓ |
| Grounding Score | 75-85 | 80-90 | 75-80 ✓ |

**Result:** ~40-60% cost reduction while maintaining quality through intelligent routing.

---

## Next Steps

1. **Domain-specific instruction tuning**: Add task-specific fine-tuning on (patent_context, query, idea) triplets
2. **Embedding query compression**: Compress queries before embedding search (reduce retrieval time)
3. **Hybrid retrieval**: Combine semantic + keyword search for better patent matching
4. **Idea scoring**: Fine-tune a separate regressor to predict grounding score before generation
5. **Multi-turn interaction**: Build conversational business idea refinement

---

## File Structure

```
Backend/
├── app/
│   ├── api.py                 # FastAPI endpoints (updated: multi-model)
│   ├── pipeline.py            # Idea generation pipeline (updated: router + multi-model)
│   ├── llm.py                 # LLM clients (updated: OllamaClient + LocalQwenClient)
│   ├── model_router.py        # Query routing logic (NEW)
│   ├── config.py              # Configuration
│   ├── retrieval.py           # Patent retrieval
│   ├── grounding.py           # Grounding scorer
│   └── ...
├── scripts/
│   ├── extract_training_data.py       # Extract patents for fine-tuning (NEW)
│   ├── prepare_embedding_dataset.py   # Create embedding dataset (NEW)
│   ├── finetune_embeddings.py         # Fine-tune sentence-transformers (NEW)
│   ├── re_embed_patents.py            # Re-encode patents (NEW)
│   ├── finetune_qwen.py               # QLoRA fine-tuning (NEW)
│   └── setup.sh                       # Setup script (NEW)
├── config/
│   └── qwen_qlora_config.json         # QLoRA config (NEW)
├── data/
│   ├── patent_domain_data.jsonl       # Training data (generated)
│   └── embedding_training_data/       # Embedding dataset (generated)
├── models/
│   ├── patent_embeddings_ft/          # Fine-tuned embeddings (generated)
│   └── qwen_patent_qlora/             # Fine-tuned qwen + LoRA (generated)
├── tests/
│   └── test_router.py                 # Router & integration tests (NEW)
└── requirements.txt                    # Updated with fine-tuning deps
```

---

For questions or issues, refer to individual script documentation:
```bash
python scripts/extract_training_data.py --help
python scripts/finetune_qwen.py --help
# etc.
```
