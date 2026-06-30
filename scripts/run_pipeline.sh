#!/usr/bin/env bash
set -euo pipefail

# Inference-only pipeline. The team does not train/fine-tune any model.
# Select one pretrained/public embedding checkpoint from Hugging Face:
#   BAAI/bge-m3
#   kietnt0603/nrk-legal-bge
#   AITeamVN/Vietnamese_Embedding

EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-AITeamVN/Vietnamese_Embedding}"

python src/bm25_retriever.py   --questions data/R2AIStage1DATA.json   --output outputs/bm25_candidates.json   --top-k 4000

python src/dense_reranker.py   --input outputs/bm25_candidates.json   --output outputs/reranked_candidates.json   --model-name "$EMBEDDING_MODEL_NAME"   --score-threshold 0.0   --top-k -1

python scripts/merge_questions_and_generate_answers.py   --questions data/R2AIStage1DATA.json   --retrieval outputs/reranked_candidates.json   --output outputs/results.json

python scripts/validate_submission.py outputs/results.json
zip -j outputs/submission.zip outputs/results.json
