# Pipeline Module

## End-to-End Flow

```
1. Load test questions
2. For each question:
   a. Hybrid retrieve (FAISS + BM25 → RRF merge)
   b. Rerank top-50 → top-10
   c. Format context with article metadata
   d. Generate answer with LLM
   e. Extract citations from answer
   f. Build relevant_docs + relevant_articles
3. Output results.json
4. Validate & zip
```

## Usage

```bash
python src/pipeline/run_pipeline.py --config configs/config.yaml
```
