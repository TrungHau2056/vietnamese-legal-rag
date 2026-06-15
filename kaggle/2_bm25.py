"""Stage 2: BM25 Indexing

INPUT:  articles.jsonl (từ Stage 1)
OUTPUT: bm25_index.pkl

Chạy trên Kaggle:
    !pip install -q rank-bm25 pyvi tqdm
    !python 2_bm25.py

Lưu ý: Tokenize 62K articles mất ~1.5h. Chạy 1 lần rồi save.
"""

import json
import os
import pickle
import sys
from collections import defaultdict
from tqdm import tqdm

from pyvi.ViTokenizer import tokenize as vi_tokenize
from rank_bm25 import BM25Okapi


ARTICLES_PATH = "articles.jsonl"
BM25_PATH = "bm25_index.pkl"


def tokenize_vi(text):
    try:
        return vi_tokenize(text).split()
    except Exception:
        return text.split()


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("STAGE 2: BM25 INDEXING")
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

    # Check existing index
    if os.path.exists(BM25_PATH):
        print(f"\n{BM25_PATH} already exists. Loading...")
        with open(BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        bm25 = bm25_data["index"]
        print(f"  BM25 corpus_size: {bm25.corpus_size}")
        if bm25.corpus_size == len(articles):
            print("  Index is up-to-date. Nothing to do.")
            return
        else:
            print(f"  Index size ({bm25.corpus_size}) != articles ({len(articles)}). Rebuilding...")

    # Build BM25 index
    print("\nBuilding BM25 index...")
    tokenized_corpus = []
    for art in tqdm(articles, desc="Tokenizing"):
        parts = [art["article_num"]]
        if art.get("law_name"):
            parts.append(art["law_name"])
        if art.get("chapter"):
            parts.append(art["chapter"])
        parts.append(art["text"])
        tokens = tokenize_vi(". ".join(parts))
        tokenized_corpus.append(tokens)

    bm25 = BM25Okapi(tokenized_corpus)
    print(f"  BM25 corpus_size: {bm25.corpus_size}")

    # Save
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"index": bm25}, f)
    print(f"  Saved to {BM25_PATH}")

    # Quick test
    test_queries = [
        "Doanh nghiệp nhỏ và vừa được hưởng những hỗ trợ gì?",
        "Điều kiện thành lập doanh nghiệp tư nhân",
    ]
    print("\n  Quick test:")
    for q in test_queries:
        tokens = tokenize_vi(q)
        scores = bm25.get_scores(tokens)
        top3 = scores.argsort()[::-1][:3]
        print(f"    Q: {q[:50]}...")
        for idx in top3:
            print(f"      → {articles[idx]['law_id']} {articles[idx]['article_num']} (score={scores[idx]:.2f})")

    print(f"\n{'=' * 60}")
    print(f"STAGE 2 COMPLETE: {BM25_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
