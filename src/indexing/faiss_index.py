"""Build and query FAISS dense index from BGE-M3 embeddings.

Reads indexes/dense.npy + indexes/metadata.jsonl → saves indexes/faiss/
"""

import json
import os

import faiss
import numpy as np
import yaml


def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_index(dense_vectors):
    """Build FAISS IndexFlatIP from dense vectors (L2-normalized → cosine)."""
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(dense_vectors)
    dim = dense_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(dense_vectors)
    return index


def main():
    cfg = load_config()
    index_dir = cfg["paths"]["indexes"]
    faiss_dir = os.path.join(index_dir, "faiss")
    os.makedirs(faiss_dir, exist_ok=True)

    print("=== FAISS Index Build ===\n")

    # Load dense vectors
    dense_path = os.path.join(index_dir, "dense.npy")
    print(f"1. Loading dense vectors from {dense_path}")
    dense_vectors = np.load(dense_path).astype("float32")
    print(f"   Shape: {dense_vectors.shape}\n")

    # Build index
    print("2. Building FAISS IndexFlatIP")
    index = build_index(dense_vectors)
    print(f"   Index size: {index.ntotal} vectors\n")

    # Save index
    index_path = os.path.join(faiss_dir, "index.faiss")
    faiss.write_index(index, index_path)
    print(f"3. Saved index to {index_path}")

    # Verify
    print("4. Verifying: query first vector")
    query = dense_vectors[0:1]
    faiss.normalize_L2(query)
    scores, ids = index.search(query, 5)
    print(f"   Top-5 IDs: {ids[0].tolist()}")
    print(f"   Top-5 scores: {scores[0].tolist()}")

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
