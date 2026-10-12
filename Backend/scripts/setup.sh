#!/bin/bash
# Setup script for Invenza backend - downloads and initializes models

set -e

echo "=========================================="
echo "Invenza Backend Setup Script"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo ""
echo "Creating directories..."
mkdir -p data/
mkdir -p models/
mkdir -p config/
mkdir -p logs/

echo "✓ Directories created"

# Download qwen model from Hugging Face (optional - for offline mode)
# This is optional since transformers will auto-download on first use
echo ""
echo "=========================================="
echo "Model Initialization"
echo "=========================================="

read -p "Download qwen2.5-3b from HuggingFace? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading qwen2.5-3b..."
    python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('qwen/qwen2.5-3b'); AutoModelForCausalLM.from_pretrained('qwen/qwen2.5-3b')"
    echo "✓ qwen2.5-3b downloaded"
else
    echo "⚠ Skipped qwen download (will be auto-downloaded on first use)"
fi

# Check Ollama
echo ""
echo "Checking Ollama models..."
read -p "Do you have ollama running? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pulling llama3.2 via Ollama..."
    ollama pull llama3.2
    echo "✓ llama3.2 ready"
else
    echo "⚠ Please install and start Ollama from https://ollama.ai"
    echo "  Then run: ollama pull llama3.2"
fi

# Test database connection
echo ""
echo "=========================================="
echo "Database Setup"
echo "=========================================="
read -p "Test PostgreSQL connection? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing database connection..."
    python -c "
import os
import psycopg
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path('.env'))
db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@db:5432/postgres')
try:
    with psycopg.connect(db_url) as conn:
        result = conn.execute('SELECT COUNT(*) FROM patents').fetchone()
        count = result[0] if result else 0
        print(f'✓ Database connected. Patents in DB: {count}')
except Exception as e:
    print(f'✗ Database connection failed: {e}')
"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Extract training data:"
echo "   python scripts/extract_training_data.py"
echo ""
echo "2. Prepare embedding dataset:"
echo "   python scripts/prepare_embedding_dataset.py"
echo ""
echo "3. Fine-tune embeddings:"
echo "   python scripts/finetune_embeddings.py"
echo ""
echo "4. Re-embed patents:"
echo "   python scripts/re_embed_patents.py"
echo ""
echo "5. Fine-tune qwen (requires GPU):"
echo "   python scripts/finetune_qwen.py"
echo ""
echo "6. Run API:"
echo "   python -m uvicorn app.api:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "=========================================="
