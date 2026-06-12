"""BM25-only retrieval + Gemini generation for fast submission.

No FAISS/dense embeddings needed. Uses BM25 + pyvi tokenization.
Optionally adds reranker if GPU available.
"""

import json
import os
import pickle
import re
import sys
import time

import yaml
from pyvi.ViTokenizer import tokenize as vi_tokenize
from rank_bm25 import BM25Okapi


def tokenize_vietnamese(text):
    try:
        return vi_tokenize(text).split()
    except Exception:
        return text.split()


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    os.chdir(project_root)

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load competition data
    with open("R2AIStage1DATA.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    # Load BM25 index
    print("Loading BM25 index...")
    with open("indexes/bm25/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    print(f"  BM25 corpus_size: {bm25.corpus_size}")

    # Load articles
    print("Loading articles...")
    articles = []
    with open("data/processed/articles.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    print(f"  {len(articles)} articles loaded")

    # BM25 top-k (retrieve more, then optional rerank)
    bm25_top_k = 100  # Retrieve broad for F2 (Recall-weighted)

    # Optional reranker
    use_reranker = "--rerank" in sys.argv
    reranker = None
    if use_reranker:
        print("Loading reranker...")
        from src.retrieval.reranker import Reranker
        reranker = Reranker(cfg)
        reranker.load()
        print("  Reranker loaded.")

    # Optional Gemini generation
    use_gemini = "--generate" in sys.argv
    generator = None
    if use_gemini:
        print("Loading Gemini generator...")
        from src.generation.generator import GeminiGenerator
        generator = GeminiGenerator(cfg)
        generator.load()
        print("  Generator loaded.")

    # Process questions
    results = []
    total = len(questions)
    start_time = time.time()
    final_top_n = 10

    for i, item in enumerate(questions):
        qid = item["id"]
        question = item["question"]

        # BM25 search
        query_tokens = tokenize_vietnamese(question)
        scores = bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1][:bm25_top_k]

        # Build candidate articles
        candidates = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            art = articles[idx]
            candidates.append({
                "law_id": art["law_id"],
                "law_name": art["law_name"],
                "article_num": art["article_num"],
                "chapter": art.get("chapter", ""),
                "text": art["text"],
                "bm25_score": float(scores[idx]),
                "bm25_rank": rank,
            })

        # Rerank if available
        if reranker and candidates:
            candidates = reranker.rerank(question, candidates, top_n=final_top_n)
        else:
            candidates = candidates[:final_top_n]

        # Generate answer if Gemini available
        answer = ""
        cited_articles = []
        if generator and candidates:
            try:
                gen_result = generator.generate(question, candidates)
                answer = gen_result["answer"]
                cited_articles = gen_result.get("relevant_articles", [])
            except Exception as e:
                print(f"  Generation error for Q{qid}: {e}")
                answer = ""

        # Build relevant_docs and relevant_articles
        # Use ALL top-N candidates for Recall (F2 weights Recall 2x)
        seen_docs = set()
        seen_arts = set()
        relevant_docs = []
        relevant_articles = []

        # Add cited articles first (from LLM answer), then fill with retrieval results
        source_articles = cited_articles if cited_articles else candidates
        for art in source_articles:
            doc_key = f"{art['law_id']}|{art['law_name']}"
            art_key = f"{art['law_id']}|{art['law_name']}|{art['article_num']}"
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                relevant_docs.append(doc_key)
            if art_key not in seen_arts:
                seen_arts.add(art_key)
                relevant_articles.append(art_key)

        # Fill from retrieval results if not already included
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
            "answer": answer,
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
        })

        # Progress
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] rate={rate:.1f} q/s, ETA={eta/60:.1f} min")

        # Checkpoint
        if (i + 1) % 200 == 0:
            with open("results_checkpoint.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  Checkpoint saved at {i+1}")

    total_time = time.time() - start_time
    print(f"\nDone! {total} questions in {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save final
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to results.json")

    # Stats
    doc_counts = [len(r["relevant_docs"]) for r in results]
    art_counts = [len(r["relevant_articles"]) for r in results]
    has_answer = sum(1 for r in results if r["answer"])
    print(f"Avg relevant_docs: {sum(doc_counts)/len(doc_counts):.1f}")
    print(f"Avg relevant_articles: {sum(art_counts)/len(art_counts):.1f}")
    print(f"Questions with answers: {has_answer}/{total}")


if __name__ == "__main__":
    main()
