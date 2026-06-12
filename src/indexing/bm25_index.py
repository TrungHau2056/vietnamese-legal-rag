"""Build and query BM25 sparse index from articles.

Tokenizes Vietnamese text with pyvi → builds BM25Okapi index.
Saves to indexes/bm25/
"""

import json
import os
import pickle

import yaml
from tqdm import tqdm


def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenize_vietnamese(text):
    """Tokenize Vietnamese text using pyvi."""
    from pyvi.ViTokenizer import tokenize
    try:
        tokenized = tokenize(text)
        return tokenized.split()
    except Exception:
        return text.split()


def main():
    cfg = load_config()
    processed_path = os.path.join(cfg["paths"]["processed_data"], "articles.jsonl")
    index_dir = cfg["paths"]["indexes"]
    bm25_dir = os.path.join(index_dir, "bm25")
    os.makedirs(bm25_dir, exist_ok=True)

    print("=== BM25 Index Build ===\n")

    # Load articles
    print(f"1. Loading articles from {processed_path}")
    articles = []
    with open(processed_path, "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    print(f"   {len(articles)} articles loaded\n")

    # Tokenize
    print("2. Tokenizing Vietnamese text")
    tokenized_corpus = []
    for art in tqdm(articles, desc="  Tokenizing"):
        # Combine article_num + law_name + chapter + text for BM25
        parts = [art["article_num"]]
        if art.get("law_name"):
            parts.append(art["law_name"])
        if art.get("chapter"):
            parts.append(art["chapter"])
        parts.append(art["text"])
        combined = ". ".join(parts)
        tokens = tokenize_vietnamese(combined)
        tokenized_corpus.append(tokens)

    # Build BM25 index
    print("\n3. Building BM25Okapi index")
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"   Corpus size: {bm25.corpus_size}\n")

    # Save
    bm25_path = os.path.join(bm25_dir, "bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"4. Saved BM25 index to {bm25_path}")

    # Save tokenized corpus for verification
    corpus_path = os.path.join(bm25_dir, "tokenized_corpus.pkl")
    with open(corpus_path, "wb") as f:
        pickle.dump(tokenized_corpus, f)

    # Verify
    print("5. Verifying: query 'doanh nghiệp nhỏ và vừa'")
    query_tokens = tokenize_vietnamese("doanh nghiệp nhỏ và vừa")
    scores = bm25.get_scores(query_tokens)
    top_ids = scores.argsort()[::-1][:5]
    for rank, idx in enumerate(top_ids):
        art = articles[idx]
        print(f"   #{rank+1}: {art['law_id']} {art['article_num']} (score={scores[idx]:.4f})")

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
