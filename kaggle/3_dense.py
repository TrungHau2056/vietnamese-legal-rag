"""Stage 3: Dense Embeddings + FAISS Index

INPUT:  articles.jsonl (từ Stage 1)
OUTPUT: dense.npy, index.faiss, metadata.jsonl

Chạy trên Kaggle (cần GPU T4/P100):
    !pip install -q sentence-transformers faiss-cpu tqdm
    !python 3_dense.py

Lưu ý: Encode 52K articles mất ~3h trên T4. Có checkpoint mỗi 5000 items.
"""

import json
import os
import sys

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


ARTICLES_PATH = "articles.jsonl"
DENSE_PATH = "dense.npy"
FAISS_PATH = "index.faiss"
META_PATH = "metadata.jsonl"
PARTIAL_PATH = "dense_partial.npy"

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 32
MAX_LENGTH = 512  # truncate at 512 tokens cho speed (BGE-M3 default 8192)
CHECKPOINT_EVERY = 5000


def build_texts(articles):
    texts = []
    for art in articles:
        parts = [art["article_num"]]
        if art.get("law_name"):
            parts.append(art["law_name"])
        if art.get("chapter"):
            parts.append(art["chapter"])
        parts.append(art["text"])
        texts.append(". ".join(parts))
    return texts


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("STAGE 3: DENSE EMBEDDINGS + FAISS")
    print("=" * 60)

    # Load articles
    if not os.path.exists(ARTICLES_PATH):
        print(f"ERROR: {ARTICLES_PATH} not found. Run Stage 1 first.")
        return

    articles = []
    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    print(f"Loaded {len(articles)} articles from {ARTICLES_PATH}")

    # ── Check if FAISS index already exists ──
    if os.path.exists(FAISS_PATH):
        print(f"\n{FAISS_PATH} already exists. Loading...")
        faiss_index = faiss.read_index(FAISS_PATH)
        print(f"  Index size: {faiss_index.ntotal} vectors, dim={faiss_index.d}")
        if faiss_index.ntotal == len(articles):
            print("  Index is complete. Skipping encoding.")
            _save_metadata(articles)
            return
        else:
            print(f"  Index size mismatch. Rebuilding...")

    # ── Check existing dense vectors ──
    if os.path.exists(DENSE_PATH):
        dense_vecs = np.load(DENSE_PATH)
        if dense_vecs.shape[0] == len(articles):
            print(f"\n{DENSE_PATH} is complete ({dense_vecs.shape}). Building FAISS...")
            _build_faiss(dense_vecs)
            _save_metadata(articles)
            return
        else:
            print(f"\n{DENSE_PATH} has wrong shape ({dense_vecs.shape[0]} vs {len(articles)}). Re-encoding...")

    # ── Encode with BGE-M3 ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"\nGPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")
    else:
        print("\nNo GPU available. Encoding on CPU (slow).")

    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
    texts = build_texts(articles)
    print(f"Encoding {len(texts)} texts (batch_size={BATCH_SIZE}, max_length={MAX_LENGTH})...")

    all_embeddings = []
    start_idx = 0

    # Resume from checkpoint
    if os.path.exists(PARTIAL_PATH):
        partial = np.load(PARTIAL_PATH)
        if partial.shape[0] < len(texts):
            start_idx = partial.shape[0]
            all_embeddings = [partial]
            print(f"  Resuming from {start_idx}/{len(texts)}")

    for i in tqdm(range(start_idx, len(texts), BATCH_SIZE), desc="Encoding"):
        batch = texts[i : i + BATCH_SIZE]
        emb = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            max_length=MAX_LENGTH,
        )
        all_embeddings.append(np.array(emb, dtype="float32"))

        # Checkpoint
        done = min(i + BATCH_SIZE, len(texts))
        if done % CHECKPOINT_EVERY < BATCH_SIZE:
            combined = np.vstack(all_embeddings)
            np.save(PARTIAL_PATH, combined)
            print(f"  Checkpoint: {combined.shape[0]}/{len(texts)}")

    # Final save
    dense_vecs = np.vstack(all_embeddings)
    np.save(DENSE_PATH, dense_vecs)
    print(f"\nDense shape: {dense_vecs.shape}")

    # Build FAISS
    _build_faiss(dense_vecs)

    # Cleanup
    del dense_vecs, model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Save metadata
    _save_metadata(articles)

    # Remove partial checkpoint
    if os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)
        print(f"Removed {PARTIAL_PATH}")

    print(f"\n{'=' * 60}")
    print(f"STAGE 3 COMPLETE: {FAISS_PATH}, {META_PATH}")
    print(f"{'=' * 60}")


def _build_faiss(dense_vecs):
    vecs = dense_vecs.astype("float32")
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, FAISS_PATH)
    print(f"  FAISS index: {index.ntotal} vectors, dim={index.d}")


def _save_metadata(articles):
    if not os.path.exists(META_PATH):
        with open(META_PATH, "w", encoding="utf-8") as f:
            for art in articles:
                f.write(
                    json.dumps(
                        {
                            "law_id": art["law_id"],
                            "law_name": art["law_name"],
                            "article_num": art["article_num"],
                            "chapter": art.get("chapter", ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"  Saved metadata to {META_PATH}")
    else:
        print(f"  {META_PATH} already exists.")


if __name__ == "__main__":
    main()
