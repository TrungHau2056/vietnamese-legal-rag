"""Encode articles with BGE-M3 and save dense embeddings.

Reads data/processed/articles.jsonl → saves embeddings to indexes/
Uses sentence-transformers (compatible with transformers 5.x).
Supports checkpointing: resumes from existing partial dense.npy.
"""

import json
import os

import numpy as np
import yaml
from tqdm import tqdm


def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_articles(processed_path):
    articles = []
    with open(processed_path, "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    return articles


def build_texts(articles):
    """Build embedding input: article_num + law_name + text."""
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
    cfg = load_config()
    processed_path = os.path.join(cfg["paths"]["processed_data"], "articles.jsonl")
    index_dir = cfg["paths"]["indexes"]
    os.makedirs(index_dir, exist_ok=True)

    print("=== BGE-M3 Embedding (sentence-transformers) ===\n")

    # Load articles
    print(f"1. Loading articles from {processed_path}")
    articles = load_articles(processed_path)
    print(f"   {len(articles)} articles loaded\n")

    # Build texts
    texts = build_texts(articles)

    # Check for existing checkpoint
    dense_path = os.path.join(index_dir, "dense.npy")
    start_idx = 0
    existing_embeddings = None
    if os.path.exists(dense_path):
        existing_embeddings = np.load(dense_path)
        if existing_embeddings.shape[0] < len(articles):
            start_idx = existing_embeddings.shape[0]
            print(f"   Found checkpoint: {start_idx}/{len(articles)} encoded\n")
        elif existing_embeddings.shape[0] == len(articles):
            print(f"   dense.npy already complete ({len(articles)} vectors). Skipping encoding.\n")
            # Still save metadata
            meta_path = os.path.join(index_dir, "metadata.jsonl")
            with open(meta_path, "w", encoding="utf-8") as f:
                for art in articles:
                    meta = {
                        "law_id": art["law_id"],
                        "law_name": art["law_name"],
                        "article_num": art["article_num"],
                        "chapter": art.get("chapter", ""),
                        "source": art.get("source", ""),
                    }
                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            print(f"   Saved metadata to {meta_path}")
            print(f"\n=== Done ===")
            return
        else:
            print(f"   dense.npy has wrong shape ({existing_embeddings.shape[0]}). Re-encoding from scratch.\n")
            existing_embeddings = None
            start_idx = 0

    # Load BGE-M3 via sentence-transformers
    model_name = cfg["embedding"]["model"]
    batch_size = cfg["embedding"].get("batch_size", 12)
    max_length = cfg["embedding"]["max_length"]

    import torch
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        device = "cuda" if gpu_mem >= 8 else "cpu"
        print(f"2. Loading {model_name} on {device} (GPU: {gpu_mem:.1f}GB)")
    else:
        device = "cpu"
        print(f"2. Loading {model_name} on {device} (no GPU available)")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    print(f"   Model loaded (max_length={max_length})\n")

    # Encode remaining texts with checkpointing every 5000 items
    remaining_texts = texts[start_idx:]
    checkpoint_every = 5000
    print(f"3. Encoding {len(remaining_texts)} texts (starting from idx {start_idx}, batch_size={batch_size}, max_length={max_length})")

    all_new_embeddings = []
    for i in tqdm(range(0, len(remaining_texts), batch_size), desc="  Encoding batches"):
        batch = remaining_texts[i : i + batch_size]
        batch_emb = model.encode(
            batch,
            batch_size=len(batch),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_new_embeddings.append(np.array(batch_emb, dtype="float32"))

        # Checkpoint
        encoded_so_far = start_idx + i + len(batch)
        if encoded_so_far % checkpoint_every < batch_size or (i + batch_size >= len(remaining_texts)):
            new_emb = np.vstack(all_new_embeddings) if all_new_embeddings else np.empty((0, 1024), dtype="float32")
            if existing_embeddings is not None:
                combined = np.vstack([existing_embeddings, new_emb])
            else:
                combined = new_emb
            np.save(dense_path, combined)
            print(f"   Checkpoint: {combined.shape[0]}/{len(articles)} saved")

    # Final save
    new_emb = np.vstack(all_new_embeddings) if all_new_embeddings else np.empty((0, 1024), dtype="float32")
    if existing_embeddings is not None:
        dense_vecs = np.vstack([existing_embeddings, new_emb])
    else:
        dense_vecs = new_emb
    print(f"\n   Dense shape: {dense_vecs.shape}\n")

    np.save(dense_path, dense_vecs)
    print(f"4. Saved dense vectors to {dense_path}")
    print(f"   Shape: {dense_vecs.shape}")

    # Save metadata
    meta_path = os.path.join(index_dir, "metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for art in articles:
            meta = {
                "law_id": art["law_id"],
                "law_name": art["law_name"],
                "article_num": art["article_num"],
                "chapter": art.get("chapter", ""),
                "source": art.get("source", ""),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"5. Saved metadata to {meta_path}")

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
