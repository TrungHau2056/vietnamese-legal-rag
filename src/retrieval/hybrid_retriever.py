"""Hybrid retrieval: dense (FAISS) + sparse (BM25) with RRF fusion.

Loads FAISS index + BM25 index + metadata → given query, returns top-K merged results.
"""

import json
import os

import faiss
import numpy as np
import yaml
from rank_bm25 import BM25Okapi

from pyvi.ViTokenizer import tokenize as vi_tokenize


def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenize_vietnamese(text):
    try:
        return vi_tokenize(text).split()
    except Exception:
        return text.split()


class HybridRetriever:
    """Hybrid retriever combining FAISS dense + BM25 sparse with RRF."""

    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.index_dir = self.cfg["paths"]["indexes"]
        self.dense_top_k = self.cfg["retrieval"]["dense_top_k"]
        self.bm25_top_k = self.cfg["retrieval"]["bm25_top_k"]
        self.rrf_k = self.cfg["retrieval"]["rrf_k"]

        self.faiss_index = None
        self.bm25 = None
        self.metadata = []
        self.model = None

    def load(self):
        """Load all indexes, model, and article texts into memory."""
        # Load metadata + article texts
        meta_path = os.path.join(self.index_dir, "metadata.jsonl")
        processed_path = os.path.join(self.cfg["paths"]["processed_data"], "articles.jsonl")
        self.metadata = []
        self.article_texts = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        with open(processed_path, "r", encoding="utf-8") as f:
            for line in f:
                self.article_texts.append(json.loads(line)["text"])

        # Load FAISS
        faiss_path = os.path.join(self.index_dir, "faiss", "index.faiss")
        self.faiss_index = faiss.read_index(faiss_path)

        # Load BM25
        import pickle
        bm25_path = os.path.join(self.index_dir, "bm25", "bm25.pkl")
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

        # Load BGE-M3 for query encoding (via sentence-transformers)
        # Keep on CPU to save VRAM for reranker
        import torch
        from sentence_transformers import SentenceTransformer

        model_name = self.cfg["embedding"]["model"]
        self.device = "cpu"
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)

        return self

    def dense_search(self, query_text, top_k=None):
        """Search FAISS dense index using BGE-M3 via sentence-transformers."""
        top_k = top_k or self.dense_top_k
        q_emb = self.model.encode([query_text], normalize_embeddings=True)
        q_vec = q_emb.astype("float32")
        faiss.normalize_L2(q_vec)
        scores, ids = self.faiss_index.search(q_vec, top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(ids[0], scores[0])):
            if idx < 0:
                continue
            results.append({"index": int(idx), "score": float(score), "rank": rank})
        return results

    def bm25_search(self, query_text, top_k=None):
        """Search BM25 sparse index."""
        top_k = top_k or self.bm25_top_k
        query_tokens = tokenize_vietnamese(query_text)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            results.append({"index": int(idx), "score": float(scores[idx]), "rank": rank})
        return results

    def rrf_merge(self, dense_results, bm25_results, k=None):
        """Merge results using Reciprocal Rank Fusion."""
        k = k or self.rrf_k
        scores = {}
        for r in dense_results:
            idx = r["index"]
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + r["rank"] + 1)
        for r in bm25_results:
            idx = r["index"]
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + r["rank"] + 1)
        # Sort by RRF score descending
        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"index": idx, "rrf_score": score} for idx, score in merged]

    def retrieve(self, query_text, top_n=None):
        """Full hybrid retrieval pipeline: dense + BM25 → RRF merge → top-N."""
        dense_results = self.dense_search(query_text)
        bm25_results = self.bm25_search(query_text)
        merged = self.rrf_merge(dense_results, bm25_results)

        top_n = top_n or 50
        top_results = merged[:top_n]

        # Attach metadata + text
        articles = []
        for r in top_results:
            idx = r["index"]
            meta = self.metadata[idx]
            articles.append({
                "law_id": meta["law_id"],
                "law_name": meta["law_name"],
                "article_num": meta["article_num"],
                "chapter": meta.get("chapter", ""),
                "text": self.article_texts[idx],
                "rrf_score": r["rrf_score"],
            })

        return articles


def main():
    cfg = load_config()
    retriever = HybridRetriever(cfg)
    retriever.load()

    # Test queries
    queries = [
        "Doanh nghiệp nhỏ và vừa được hưởng những hỗ trợ gì?",
        "Điều kiện thành lập doanh nghiệp tư nhân",
        "Quyền và nghĩa vụ của người lao động",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        results = retriever.retrieve(q, top_n=10)
        for i, r in enumerate(results):
            print(f"  #{i+1}: {r['law_id']} {r['article_num']} (RRF={r['rrf_score']:.4f})")


if __name__ == "__main__":
    main()
