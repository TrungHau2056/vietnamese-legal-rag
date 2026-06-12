# Indexing Module

## BGE-M3 Embedding

- Model: `BAAI/bge-m3` (568M params, 8192 ctx)
- Output: dense vectors (1024 dim) + sparse vectors (SPLADE-style)
- Use `FlagEmbedding` library: `BGEM3FlagModel`
- Batch processing on GPU, encode with `encode()` → dense + sparse + colbert vectors

## FAISS Index

- Dense vectors → FAISS IndexFlatIP (inner product = cosine similarity after normalization)
- Store metadata mapping: index → (law_id, article_num, text)
- Save/load with `faiss.write_index()` / `faiss.read_index()`

## BM25 Index

- Tokenize with `underthesea.word_tokenize()` or `pyvi.ViTokenizer.tokenize()`
- Build with `rank_bm25.BM25Okapi`
- Catches exact law IDs (e.g., "04/2017/QH14"), article numbers, specific legal terms
