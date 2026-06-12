# Retrieval Module

## Hybrid Retrieval (RRF Fusion)

1. Dense retrieval: top-K from FAISS (cosine similarity)
2. Sparse retrieval: top-K from BM25
3. Merge with Reciprocal Rank Fusion: `score(d) = Σ 1/(k + rank_i(d))`, k=60

Default params:
- dense_top_k: 50
- bm25_top_k: 50
- rrf_k: 60

## Cross-Encoder Reranking

- Model: `BAAI/bge-reranker-v2-m3` (568M, 8192 ctx)
- Input: (query, article_text) pairs → relevance score
- Rerank top-50 hybrid → top-10

## F2 Optimization

- F2 weights Recall 2x → missing relevant articles is costly
- Strategy: retrieve broadly (top-100), rerank aggressively
- Consider: increase bm25_top_k for law-ID-heavy queries
