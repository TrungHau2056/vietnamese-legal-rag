"""Cross-encoder reranking with BGE-reranker-v2-m3.

Takes top-N hybrid results → reranks → returns top-K with relevance scores.
Uses transformers directly to avoid FlagEmbedding compatibility issues.
"""

import json
import os

import yaml


def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Reranker:
    """Cross-encoder reranker using BGE-reranker-v2-m3 via transformers."""

    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.model_name = self.cfg["reranker"]["model"]
        self.top_n = self.cfg["reranker"]["top_n"]
        self.tokenizer = None
        self.model = None

    def load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        return self

    def rerank(self, query, articles, top_n=None):
        """Rerank articles by relevance to query.

        Args:
            query: Question text
            articles: List of dicts with 'text' field
            top_n: Number of top results to return

        Returns:
            List of articles sorted by reranker score (descending)
        """
        import torch

        top_n = top_n or self.top_n
        if not articles:
            return []

        pairs = [[query, art["text"]] for art in articles]

        with torch.no_grad():
            features = self.tokenizer(
                pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            features = {k: v.to(self.device) for k, v in features.items()}
            scores = self.model(**features).logits.squeeze(-1).float().tolist()

        if isinstance(scores, float):
            scores = [scores]

        for art, score in zip(articles, scores):
            art["rerank_score"] = float(score)

        ranked = sorted(articles, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_n]


def main():
    cfg = load_config()
    reranker = Reranker(cfg)
    reranker.load()

    query = "Doanh nghiệp nhỏ và vừa được hưởng những hỗ trợ gì?"
    test_articles = [
        {"law_id": "04/2017/QH14", "article_num": "Điều 13", "text": "Điều 13. Hỗ trợ về công nghệ. Doanh nghiệp nhỏ và vừa được hỗ trợ chuyển giao công nghệ."},
        {"law_id": "59/2020/QH14", "article_num": "Điều 17", "text": "Điều 17. Góp vốn, mua cổ phần, mua phần vốn góp."},
        {"law_id": "04/2017/QH14", "article_num": "Điều 12", "text": "Điều 12. Hỗ trợ về mặt bằng sản xuất, kinh doanh. Doanh nghiệp nhỏ và vừa được ưu tiên thuê mặt bằng."},
    ]

    results = reranker.rerank(query, test_articles, top_n=3)
    for r in results:
        print(f"  {r['law_id']} {r['article_num']} score={r['rerank_score']:.4f}")


if __name__ == "__main__":
    main()
