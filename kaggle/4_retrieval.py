"""Stage 4: Hybrid Retrieval + Reranking + Submission

INPUT:  articles.jsonl, bm25_index.pkl, index.faiss, metadata.jsonl,
        R2AIStage1DATA.json (competition questions)
OUTPUT: results.json, submission.zip

Chạy trên Kaggle (cần GPU T4/P100 cho reranker):
    !pip install -q sentence-transformers rank-bm25 pyvi faiss-cpu tqdm
    !python 4_retrieval.py

Option flags:
    --no-rerank     : Skip reranker (nhanh hơn, kém chất lượng hơn)
    --bm25-only     : Chỉ dùng BM25 (không cần FAISS)
    --limit N       : Chỉ chạy N câu hỏi đầu tiên (debug)
"""

import json
import os
import pickle
import re
import sys
import time
import zipfile
import glob
from collections import defaultdict

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize as vi_tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm


# ── Config ──────────────────────────────────────────────────────────────
ARTICLES_PATH = "articles.jsonl"
BM25_PATH = "bm25_index.pkl"
FAISS_PATH = "index.faiss"
META_PATH = "metadata.jsonl"
RESULTS_PATH = "results.json"

MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

DENSE_TOP_K = 50
BM25_TOP_K = 100
RRF_K = 60
RERANK_TOP_N = 20
FINAL_TOP_N = 10

# Parse flags
NO_RERANK = "--no-rerank" in sys.argv
BM25_ONLY = "--bm25-only" in sys.argv
LIMIT = None
for i, arg in enumerate(sys.argv):
    if arg == "--limit" and i + 1 < len(sys.argv):
        LIMIT = int(sys.argv[i + 1])


def tokenize_vi(text):
    try:
        return vi_tokenize(text).split()
    except Exception:
        return text.split()


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("STAGE 4: RETRIEVAL + SUBMISSION")
    print("=" * 60)

    mode = "BM25-ONLY" if BM25_ONLY else ("HYBRID+RERANK" if not NO_RERANK else "HYBRID-NO-RERANK")
    print(f"  Mode: {mode}")
    if LIMIT:
        print(f"  Limit: {LIMIT} questions")

    # ── Load articles ──
    if not os.path.exists(ARTICLES_PATH):
        print(f"ERROR: {ARTICLES_PATH} not found. Run Stage 1 first.")
        return
    articles = []
    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    print(f"  Loaded {len(articles)} articles")

    # ── Load metadata ──
    if not os.path.exists(META_PATH):
        print(f"ERROR: {META_PATH} not found. Run Stage 3 first.")
        return
    metadata = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    print(f"  Loaded {len(metadata)} metadata entries")

    # ── Load BM25 ──
    if not os.path.exists(BM25_PATH):
        print(f"ERROR: {BM25_PATH} not found. Run Stage 2 first.")
        return
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)["index"]
    print(f"  BM25 corpus_size: {bm25.corpus_size}")

    # ── Load FAISS (optional) ──
    faiss_index = None
    query_model = None
    if not BM25_ONLY:
        if not os.path.exists(FAISS_PATH):
            print(f"  WARNING: {FAISS_PATH} not found. Falling back to BM25-only.")
            BM25_ONLY = True
        else:
            faiss_index = faiss.read_index(FAISS_PATH)
            print(f"  FAISS index: {faiss_index.ntotal} vectors, dim={faiss_index.d}")

            # Load query encoder on GPU (preferred) or CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Loading query encoder on {device}...")
            query_model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)

    # ── Load reranker (optional) ──
    reranker_model = None
    reranker_tokenizer = None
    use_reranker = False

    if not NO_RERANK and not BM25_ONLY and torch.cuda.is_available():
        print("  Loading reranker on GPU...")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
        reranker_model.to("cuda")
        reranker_model.eval()
        use_reranker = True
        print("  Reranker loaded.")
    elif not NO_RERANK:
        print("  No GPU available. Skipping reranker.")

    # ── Load competition questions ──
    questions = _load_questions()
    if not questions:
        return
    if LIMIT:
        questions = questions[:LIMIT]
        print(f"  Running {len(questions)} questions (limited)")

    # ── Retrieval functions ──
    def dense_search(query_text, top_k=DENSE_TOP_K):
        q_emb = query_model.encode([query_text], normalize_embeddings=True)
        q_vec = q_emb.astype("float32")
        faiss.normalize_L2(q_vec)
        scores, ids = faiss_index.search(q_vec, top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(ids[0], scores[0])):
            if idx < 0:
                continue
            results.append({"index": int(idx), "score": float(score), "rank": rank})
        return results

    def bm25_search(query_text, top_k=BM25_TOP_K):
        tokens = tokenize_vi(query_text)
        scores = bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            results.append({"index": int(idx), "score": float(scores[idx]), "rank": rank})
        return results

    def rrf_merge(dense_results, bm25_results, k=RRF_K):
        scores = {}
        for r in dense_results:
            scores[r["index"]] = scores.get(r["index"], 0) + 1.0 / (k + r["rank"] + 1)
        for r in bm25_results:
            scores[r["index"]] = scores.get(r["index"], 0) + 1.0 / (k + r["rank"] + 1)
        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"index": idx, "rrf_score": score} for idx, score in merged]

    def rerank(query, candidates, top_n=FINAL_TOP_N):
        pairs = [[query, art["text"]] for art in candidates]
        with torch.no_grad():
            features = reranker_tokenizer(
                pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            features = {k: v.to("cuda") for k, v in features.items()}
            scores = reranker_model(**features).logits.squeeze(-1).float().tolist()
        if isinstance(scores, float):
            scores = [scores]
        for art, score in zip(candidates, scores):
            art["rerank_score"] = float(score)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]

    # ── Run retrieval ──
    print(f"\n  Processing {len(questions)} questions...")
    results = []
    total = len(questions)
    start_time = time.time()

    for i, item in enumerate(questions):
        qid = item["id"]
        question = item["question"]

        # Hybrid or BM25-only retrieval
        if BM25_ONLY:
            bm25_res = bm25_search(question)
            candidates = []
            for r in bm25_res[:30]:
                idx = r["index"]
                meta = metadata[idx]
                candidates.append({
                    "law_id": meta["law_id"],
                    "law_name": meta["law_name"],
                    "article_num": meta["article_num"],
                    "chapter": meta.get("chapter", ""),
                    "text": articles[idx]["text"],
                    "bm25_score": r["score"],
                })
        else:
            dense_res = dense_search(question)
            bm25_res = bm25_search(question)
            merged = rrf_merge(dense_res, bm25_res)
            candidates = []
            for r in merged[:30]:
                idx = r["index"]
                meta = metadata[idx]
                candidates.append({
                    "law_id": meta["law_id"],
                    "law_name": meta["law_name"],
                    "article_num": meta["article_num"],
                    "chapter": meta.get("chapter", ""),
                    "text": articles[idx]["text"],
                    "rrf_score": r["rrf_score"],
                })

        # Rerank
        if use_reranker and candidates:
            try:
                candidates = rerank(question, candidates[:RERANK_TOP_N], top_n=FINAL_TOP_N)
            except Exception:
                candidates = candidates[:FINAL_TOP_N]
        else:
            candidates = candidates[:FINAL_TOP_N]

        # Build output
        seen_docs, seen_arts = set(), set()
        relevant_docs, relevant_articles = [], []
        for art in candidates:
            doc_key = f"{art['law_id']}|{art['law_name']}"
            art_key = f"{art['law_id']}|{art['law_name']}|{art['article_num']}"
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                relevant_docs.append(doc_key)
            if art_key not in seen_arts:
                seen_arts.add(art_key)
                relevant_articles.append(art_key)

        results.append({
            "id": qid,
            "question": question,
            "answer": "",
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"    [{i+1}/{total}] rate={rate:.1f} q/s, ETA={eta/60:.1f} min")

        if (i + 1) % 200 == 0:
            with open("results_checkpoint.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    print(f"\n  Done! {total} questions in {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── Save results ──
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved {RESULTS_PATH} ({len(results)} entries)")

    # ── Create submission zip ──
    with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(RESULTS_PATH)
    print("  Created submission.zip")

    # ── Stats ──
    doc_counts = [len(r["relevant_docs"]) for r in results]
    art_counts = [len(r["relevant_articles"]) for r in results]
    print(f"\n  Avg relevant_docs: {sum(doc_counts)/len(doc_counts):.1f}")
    print(f"  Avg relevant_articles: {sum(art_counts)/len(art_counts):.1f}")

    # ── Validate ──
    errors = _validate(results)
    if errors:
        print(f"\n  Validation errors: {len(errors)}")
        for e in errors[:10]:
            print(f"    - {e}")
    else:
        print("\n  Validation: ALL PASSED")

    # ── Check for slug law_ids in output ──
    slug_arts = 0
    for r in results:
        for art_key in r["relevant_articles"]:
            law_id = art_key.split("|")[0]
            if not re.match(r"^\d{1,3}/\d{4}/", law_id):
                slug_arts += 1
                break
    if slug_arts > 0:
        print(f"\n  WARNING: {slug_arts} questions have slug-format law_ids in output!")
    else:
        print("\n  All law_ids in output are standard format.")

    print(f"\n{'=' * 60}")
    print(f"STAGE 4 COMPLETE: {RESULTS_PATH}, submission.zip")
    print(f"{'=' * 60}")


def _load_questions():
    search_paths = [
        "R2AIStage1DATA.json",
        "/kaggle/input/r2ai-stage1/R2AIStage1DATA.json",
        "/kaggle/input/r2ai2026/R2AIStage1DATA.json",
    ]
    input_path = None
    for p in search_paths:
        if os.path.exists(p):
            input_path = p
            break
    if not input_path:
        matches = glob.glob("/kaggle/input/**/R2AIStage1DATA.json", recursive=True)
        if matches:
            input_path = matches[0]

    if not input_path:
        print("  ERROR: R2AIStage1DATA.json not found!")
        print("  Upload it as a Kaggle dataset.")
        return None

    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"  Loaded {len(questions)} questions from {input_path}")
    return questions


def _validate(results):
    errors = []
    for i, r in enumerate(results):
        for field in ["id", "question", "answer", "relevant_docs", "relevant_articles"]:
            if field not in r:
                errors.append(f"Entry {i}: missing field '{field}'")
        for doc in r.get("relevant_docs", []):
            if "|" not in doc:
                errors.append(f"Entry {i}: relevant_docs missing '|' separator: {doc[:50]}")
        for art in r.get("relevant_articles", []):
            parts = art.split("|")
            if len(parts) < 3:
                errors.append(f"Entry {i}: relevant_articles need law_id|law_name|article_num: {art[:50]}")
    return errors


if __name__ == "__main__":
    main()
