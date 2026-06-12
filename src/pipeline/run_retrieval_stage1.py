"""Run retrieval on R2AI Stage 1 competition data.

Reads R2AIStage1DATA.json, runs hybrid retrieval + reranking for each question,
produces results.json with empty answer but filled relevant_docs and relevant_articles.

Output format per question:
{
  "id": int,
  "question": str,
  "answer": "",
  "relevant_docs": ["law_id|law_name", ...],
  "relevant_articles": ["law_id|law_name|article_num", ...]
}
"""

import json
import os
import sys
import time

import yaml


def log(msg):
    print(msg, flush=True)


def main():
    # Fix import path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    os.chdir(project_root)

    # Fix Windows encoding
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Load config
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load competition data
    input_path = "R2AIStage1DATA.json"
    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    log(f"Loaded {len(questions)} questions from {input_path}")

    # Load retriever + reranker
    log("Loading retriever...")
    from src.retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(cfg)
    retriever.load()
    log("Retriever loaded.")

    # Optionally skip reranker for speed (--no-rerank flag)
    use_reranker = "--no-rerank" not in sys.argv
    reranker = None
    if use_reranker:
        log("Loading reranker...")
        from src.retrieval.reranker import Reranker
        reranker = Reranker(cfg)
        reranker.load()
        log("Reranker loaded. Starting retrieval...")
    else:
        log("Skipping reranker (--no-rerank). Using hybrid RRF only.")

    # Process each question
    results = []
    total = len(questions)
    start_time = time.time()

    for i, item in enumerate(questions):
        qid = item["id"]
        question = item["question"]

        # Retrieve top candidates via hybrid, then rerank top subset
        candidates = retriever.retrieve(question, top_n=50)

        if reranker:
            # Rerank top-10 only (GPU: fast enough, CPU: too slow for more)
            reranked = reranker.rerank(question, candidates[:10], top_n=10)
        else:
            # No reranker: use top-10 from hybrid RRF directly
            reranked = candidates[:10]

        # Build relevant_docs (unique law_id|law_name)
        seen_docs = set()
        relevant_docs = []
        for art in reranked:
            doc_key = f"{art['law_id']}|{art['law_name']}"
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                relevant_docs.append(doc_key)

        # Build relevant_articles (law_id|law_name|article_num)
        seen_articles = set()
        relevant_articles = []
        for art in reranked:
            art_key = f"{art['law_id']}|{art['law_name']}|{art['article_num']}"
            if art_key not in seen_articles:
                seen_articles.add(art_key)
                relevant_articles.append(art_key)

        results.append({
            "id": qid,
            "question": question,
            "answer": "",
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
        })

        # Progress logging every 10 questions
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            log(f"  [{i+1}/{total}] rate={rate:.1f} q/s, ETA={eta/60:.1f} min")

        # Save checkpoint every 100 questions
        if (i + 1) % 100 == 0:
            with open("results_checkpoint.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            log(f"  Checkpoint saved at {i+1}")

    total_time = time.time() - start_time
    log(f"\nDone! {total} questions in {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save final results
    output_path = "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"Saved to {output_path}")

    # Quick stats
    doc_counts = [len(r["relevant_docs"]) for r in results]
    art_counts = [len(r["relevant_articles"]) for r in results]
    log(f"Avg relevant_docs: {sum(doc_counts)/len(doc_counts):.1f}")
    log(f"Avg relevant_articles: {sum(art_counts)/len(art_counts):.1f}")


if __name__ == "__main__":
    main()
