"""Kaggle end-to-end pipeline for Vietnamese Legal RAG.

Run this as a Kaggle notebook cell or standalone script.
Assumes the repo is cloned to /kaggle/working/vietnamese-legal-rag.

Usage on Kaggle:
    !git clone https://github.com/TrungHau2056/vietnamese-legal-rag.git
    %cd vietnamese-legal-rag
    !pip install -q sentence-transformers rank-bm25 pyvi faiss-cpu google-genai
    !python kaggle_pipeline.py

Set GEMINI_API_KEY as a Kaggle secret:
    from kaggle_secrets import UserSecretsClient
    os.environ['GEMINI_API_KEY'] = UserSecretsClient().get_secret('GEMINI_API_KEY')
"""

import json
import os
import shutil
import subprocess
import sys
import time


def setup_env():
    """Set up environment and install dependencies."""
    print("=== Setting up environment ===\n")

    # Fix encoding
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Install deps if missing
    deps = ["sentence-transformers", "rank-bm25", "pyvi", "faiss-cpu",
            "google-genai", "pyyaml", "tqdm", "datasets"]
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            print(f"  Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])

    print("  Dependencies OK\n")


def step1_data_collection():
    """Download and process legal documents from HuggingFace."""
    print("\n" + "=" * 60)
    print("STEP 1: DATA COLLECTION & PROCESSING")
    print("=" * 60 + "\n")

    processed_path = "data/processed/articles.jsonl"
    if os.path.exists(processed_path):
        count = sum(1 for _ in open(processed_path, "r", encoding="utf-8"))
        print(f"  articles.jsonl exists ({count} articles). Skipping.\n")
        return

    # Download from HuggingFace
    print("  Loading HuggingFace datasets...")
    from src.data_collection.load_hf_datasets import load_phapdien, load_config
    cfg = load_config()
    os.makedirs("data/raw/phapdien", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Load phapdien (primary corpus, already chunked by Điều)
    phapdien_docs = load_phapdien(cfg)
    print(f"  phapdien: {len(phapdien_docs)} raw rows")

    # Process phapdien directly
    print("  Processing phapdien into articles...")
    from src.processing.chunker import process_phapdien, deduplicate_articles
    articles = process_phapdien(cfg)
    print(f"  After processing: {len(articles)} articles")

    # Deduplicate
    deduped = deduplicate_articles(articles)
    print(f"  After dedup: {len(deduped)} articles")

    # Save
    with open(processed_path, "w", encoding="utf-8") as f:
        for art in deduped:
            f.write(json.dumps(art, ensure_ascii=False) + "\n")
    print(f"  Saved to {processed_path}")

    # Summary
    law_counts = {}
    for art in deduped:
        law_counts[art["law_id"]] = law_counts.get(art["law_id"], 0) + 1
    print(f"  Unique laws: {len(law_counts)}")
    print(f"  Total articles: {len(deduped)}\n")


def step2_indexing():
    """Build BM25 + dense embeddings + FAISS indexes."""
    print("\n" + "=" * 60)
    print("STEP 2: INDEXING")
    print("=" * 60 + "\n")

    # BM25
    bm25_path = "indexes/bm25/bm25.pkl"
    if not os.path.exists(bm25_path):
        print("  Building BM25 index...")
        from src.indexing.bm25_index import main as bm25_main
        bm25_main()
    else:
        print("  BM25 index exists. Skipping.\n")

    # Dense embeddings
    dense_path = "indexes/dense.npy"
    if not os.path.exists(dense_path):
        print("  Building BGE-M3 dense embeddings...")
        from src.indexing.embedder import main as embed_main
        embed_main()
    else:
        print("  Dense embeddings exist. Skipping.\n")

    # FAISS
    faiss_path = "indexes/faiss/index.faiss"
    if not os.path.exists(faiss_path):
        print("  Building FAISS index...")
        from src.indexing.faiss_index import main as faiss_main
        faiss_main()
    else:
        print("  FAISS index exists. Skipping.\n")


def step3_retrieval_and_generation():
    """Run hybrid retrieval + Gemini generation on competition questions."""
    print("\n" + "=" * 60)
    print("STEP 3: RETRIEVAL + GENERATION")
    print("=" * 60 + "\n")

    # Check for Gemini API key
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        # Try loading from .env
        env_path = ".env"
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.strip().startswith("GEMINI_API_KEY="):
                        api_key = line.strip().split("=", 1)[1]
                        os.environ["GEMINI_API_KEY"] = api_key
                        break

    use_gemini = bool(api_key)
    if use_gemini:
        print("  Gemini API key found. Will generate answers.")
    else:
        print("  No GEMINI_API_KEY. Will do retrieval only (empty answers).")

    # Load competition data (check multiple locations for Kaggle)
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
        # Try globbing /kaggle/input/
        import glob
        matches = glob.glob("/kaggle/input/**/R2AIStage1DATA.json", recursive=True)
        if matches:
            input_path = matches[0]

    if not input_path:
        print("  ERROR: R2AIStage1DATA.json not found!")
        print("  Upload it as a Kaggle dataset and re-run.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"  Loaded {len(questions)} questions\n")

    # Load retriever
    import yaml
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Decide: hybrid or BM25-only
    faiss_path = "indexes/faiss/index.faiss"
    use_hybrid = os.path.exists(faiss_path)

    if use_hybrid:
        print("  Using hybrid retrieval (dense + BM25)...")
        from src.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(cfg)
        retriever.load()
    else:
        print("  Using BM25-only retrieval (no FAISS index)...")
        import pickle
        from pyvi.ViTokenizer import tokenize as vi_tokenize
        from rank_bm25 import BM25Okapi

        with open("indexes/bm25/bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        print(f"  BM25 corpus_size: {bm25.corpus_size}")

        articles = []
        with open("data/processed/articles.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))
        print(f"  {len(articles)} articles loaded")

    # Load reranker (GPU available on Kaggle)
    use_reranker = False
    try:
        import torch
        if torch.cuda.is_available():
            print("  Loading reranker on GPU...")
            from src.retrieval.reranker import Reranker
            reranker = Reranker(cfg)
            reranker.load()
            use_reranker = True
            print("  Reranker loaded.")
        else:
            print("  No GPU, skipping reranker.")
    except Exception as e:
        print(f"  Reranker failed: {e}. Skipping.")

    # Load generator
    generator = None
    if use_gemini:
        try:
            from src.generation.generator import GeminiGenerator
            generator = GeminiGenerator(cfg)
            generator.load()
            print("  Generator loaded.\n")
        except Exception as e:
            print(f"  Generator failed: {e}. Will use empty answers.\n")
            generator = None

    # Process questions
    results = []
    total = len(questions)
    start_time = time.time()
    bm25_top_k = 100
    final_top_n = 10

    def tokenize_vietnamese(text):
        try:
            return vi_tokenize(text).split()
        except Exception:
            return text.split()

    for i, item in enumerate(questions):
        qid = item["id"]
        question = item["question"]

        # Retrieve
        if use_hybrid:
            candidates = retriever.retrieve(question, top_n=50)
        else:
            query_tokens = tokenize_vietnamese(question)
            scores = bm25.get_scores(query_tokens)
            top_indices = scores.argsort()[::-1][:bm25_top_k]
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
                })

        # Rerank
        if use_reranker and candidates:
            try:
                candidates = reranker.rerank(question, candidates[:20], top_n=final_top_n)
            except Exception:
                candidates = candidates[:final_top_n]
        else:
            candidates = candidates[:final_top_n]

        # Generate answer
        answer = ""
        cited_articles = []
        if generator and candidates:
            try:
                gen_result = generator.generate(question, candidates)
                answer = gen_result["answer"]
                cited_articles = gen_result.get("relevant_articles", [])
            except Exception as e:
                print(f"  Gen error Q{qid}: {e}")
                answer = ""

        # Build output
        seen_docs = set()
        seen_arts = set()
        relevant_docs = []
        relevant_articles = []

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

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] rate={rate:.1f} q/s, ETA={eta/60:.1f} min")

        if (i + 1) % 200 == 0:
            with open("results_checkpoint.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    print(f"\n  Done! {total} questions in {total_time:.1f}s ({total_time/60:.1f} min)")

    return results


def step4_submission(results):
    """Save results.json and create submission zip."""
    print("\n" + "=" * 60)
    print("STEP 4: SUBMISSION")
    print("=" * 60 + "\n")

    # Save results
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved results.json ({len(results)} entries)")

    # Stats
    doc_counts = [len(r["relevant_docs"]) for r in results]
    art_counts = [len(r["relevant_articles"]) for r in results]
    has_answer = sum(1 for r in results if r["answer"])
    print(f"  Avg relevant_docs: {sum(doc_counts)/len(doc_counts):.1f}")
    print(f"  Avg relevant_articles: {sum(art_counts)/len(art_counts):.1f}")
    print(f"  Questions with answers: {has_answer}/{len(results)}")

    # Create zip
    import zipfile
    zip_path = "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("results.json")
    print(f"  Created {zip_path}")

    # Validate format
    errors = validate_results(results)
    if errors:
        print(f"\n  Validation errors:")
        for err in errors[:10]:
            print(f"    - {err}")
    else:
        print(f"\n  Validation: ALL PASSED")


def validate_results(results):
    """Validate results.json format."""
    errors = []
    required_fields = ["id", "question", "answer", "relevant_docs", "relevant_articles"]
    for i, r in enumerate(results):
        for field in required_fields:
            if field not in r:
                errors.append(f"Entry {i}: missing field '{field}'")

        # Check law_name format
        for doc in r.get("relevant_docs", []):
            if "|" not in doc:
                errors.append(f"Entry {i}: relevant_docs missing '|' separator: {doc[:50]}")

        for art in r.get("relevant_articles", []):
            parts = art.split("|")
            if len(parts) < 3:
                errors.append(f"Entry {i}: relevant_articles need law_id|law_name|article_num: {art[:50]}")

    return errors


def main():
    t0 = time.time()

    # Ensure we're in the project root
    if not os.path.exists("configs/config.yaml"):
        print("ERROR: Run this script from the project root directory.")
        sys.exit(1)

    setup_env()
    step1_data_collection()
    step2_indexing()
    results = step3_retrieval_and_generation()
    if results:
        step4_submission(results)

    print(f"\n{'=' * 60}")
    print(f"TOTAL TIME: {(time.time() - t0) / 60:.1f} minutes")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
