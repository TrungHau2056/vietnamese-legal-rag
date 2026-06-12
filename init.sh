#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON=".venv/Scripts/python.exe"

echo "==> Working directory: $PWD"

# Create venv if not exists
if [ ! -f ".venv/Scripts/python.exe" ]; then
    echo "==> Creating .venv"
    /c/Users/hautt/AppData/Local/Programs/Python/Python310/python.exe -m venv .venv
fi

echo "==> Checking Python"
"$PYTHON" --version

echo "==> Installing dependencies"
"$PYTHON" -m pip install -r requirements.txt --quiet

echo "==> Verifying data pipeline (smoke test)"
if [ -f "data/processed/articles.jsonl" ]; then
    echo "    articles.jsonl exists"
    LINE_COUNT=$(wc -l < "data/processed/articles.jsonl")
    echo "    Lines: $LINE_COUNT"
else
    echo "    articles.jsonl NOT found — run: $PYTHON src/processing/chunker.py"
fi

echo "==> Verifying indexes"
if [ -d "indexes/faiss" ] && [ "$(ls -A indexes/faiss 2>/dev/null)" ]; then
    echo "    FAISS index exists"
else
    echo "    FAISS index NOT found — run indexing pipeline"
fi

if [ -d "indexes/bm25" ] && [ "$(ls -A indexes/bm25 2>/dev/null)" ]; then
    echo "    BM25 index exists"
else
    echo "    BM25 index NOT found — run indexing pipeline"
fi

echo ""
echo "Commands:"
echo "  Load datasets:   .venv/Scripts/python.exe src/data_collection/load_hf_datasets.py"
echo "  Process & chunk: .venv/Scripts/python.exe src/processing/chunker.py"
echo "  Full pipeline:   .venv/Scripts/python.exe src/pipeline/run_pipeline.py --config configs/config.yaml"
