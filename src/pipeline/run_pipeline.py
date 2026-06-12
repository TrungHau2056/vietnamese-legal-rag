"""End-to-end pipeline: indexing → retrieval → generation → submission.

Usage:
    python src/pipeline/run_pipeline.py --config configs/config.yaml
    python src/pipeline/run_pipeline.py --step indexing
    python src/pipeline/run_pipeline.py --step retrieval
    python src/pipeline/run_pipeline.py --step generation
    python src/pipeline/run_pipeline.py --step full
"""

import argparse
import json
import os
import sys
import time

import yaml


def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_indexing(cfg):
    """Step 1: Build embeddings + FAISS + BM25 indexes."""
    print("\n" + "=" * 60)
    print("STEP 1: INDEXING")
    print("=" * 60)

    index_dir = cfg["paths"]["indexes"]
    faiss_path = os.path.join(index_dir, "faiss", "index.faiss")
    bm25_path = os.path.join(index_dir, "bm25", "bm25.pkl")

    # Step 1a: Embed
    dense_path = os.path.join(index_dir, "dense.npy")
    if not os.path.exists(dense_path):
        print("\n1a. Running BGE-M3 embedder...")
        t0 = time.time()
        from indexing.embedder import main as embed_main
        # Change working context
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from indexing.embedder import main as run_embed
        run_embed()
        print(f"   Embedding done in {time.time()-t0:.1f}s")
    else:
        print(f"\n1a. Dense embeddings already exist: {dense_path}")

    # Step 1b: FAISS index
    if not os.path.exists(faiss_path):
        print("\n1b. Building FAISS index...")
        from indexing.faiss_index import main as run_faiss
        run_faiss()
    else:
        print(f"\n1b. FAISS index already exists: {faiss_path}")

    # Step 1c: BM25 index
    if not os.path.exists(bm25_path):
        print("\n1c. Building BM25 index...")
        from indexing.bm25_index import main as run_bm25
        run_bm25()
    else:
        print(f"\n1c. BM25 index already exists: {bm25_path}")


def run_retrieval_test(cfg):
    """Step 2: Test hybrid retrieval + reranking on sample queries."""
    print("\n" + "=" * 60)
    print("STEP 2: RETRIEVAL TEST")
    print("=" * 60)

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(cfg)
    retriever.load()

    queries = [
        "Doanh nghiệp nhỏ và vừa được hưởng những hỗ trợ gì?",
        "Điều kiện thành lập doanh nghiệp",
        "Quyền và nghĩa vụ của người lao động theo Bộ luật Lao động",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        results = retriever.retrieve(q, top_n=10)
        for i, r in enumerate(results):
            print(f"  #{i+1}: {r['law_id']} {r['article_num']} (RRF={r['rrf_score']:.4f})")


def run_generation_test(cfg):
    """Step 3: Test LLM generation on a sample question."""
    print("\n" + "=" * 60)
    print("STEP 3: GENERATION TEST")
    print("=" * 60)

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.reranker import Reranker
    from generation.generator import GeminiGenerator

    # Load retrieval
    retriever = HybridRetriever(cfg)
    retriever.load()

    # Load reranker
    reranker = Reranker(cfg)
    reranker.load()

    # Load generator
    generator = GeminiGenerator(cfg)
    generator.load()

    # Test
    question = "Doanh nghiệp nhỏ và vừa được hưởng những hỗ trợ gì theo quy định của pháp luật?"
    print(f"\nQuestion: {question}")

    # Retrieve
    results = retriever.retrieve(question, top_n=50)
    print(f"Retrieved {len(results)} articles via hybrid search")

    # Rerank
    reranked = reranker.rerank(question, results, top_n=10)
    print(f"Reranked to top-10")

    # Generate
    result = generator.generate(question, reranked)
    print(f"\nAnswer:\n{result['answer']}\n")
    print(f"Relevant docs: {result['relevant_docs']}")
    print(f"Relevant articles: {[a['article_num'] for a in result['relevant_articles']]}")


def main():
    parser = argparse.ArgumentParser(description="Vietnamese Legal RAG Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--step", default="full",
                        choices=["indexing", "retrieval", "generation", "full"],
                        help="Which step to run")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Add src/ to Python path
    src_dir = os.path.dirname(os.path.dirname(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    if args.step in ("indexing", "full"):
        run_indexing(cfg)

    if args.step in ("retrieval", "full"):
        run_retrieval_test(cfg)

    if args.step in ("generation", "full"):
        run_generation_test(cfg)

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
