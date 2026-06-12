"""LLM answer generation using Google Gemini API.

Given a question + retrieved articles → generate Vietnamese legal answer with citations.
"""

import json
import os
import re

import yaml


def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


SYSTEM_PROMPT = """Bạn là trợ lý pháp lý chuyên về luật Việt Nam cho doanh nghiệp nhỏ và vừa (SME).
Nhiệm vụ: Trả lời câu hỏi pháp lý dựa trên các điều luật được cung cấp.

Yêu cầu bắt buộc:
1. Trích dẫn đúng điều luật: "Theo Điều X [Loại văn bản Mã văn bản Trích yếu]..."
2. Chỉ sử dụng thông tin từ các điều luật được cung cấp, không tự bịa
3. Trả lời đầy đủ, chính xác, rõ ràng
4. Nếu không đủ thông tin, ghi rõ "Căn cứ vào các điều luật được cung cấp, chưa đủ thông tin để trả lời đầy đủ"
5. Luôn nêu căn cứ pháp lý đầu tiên rồi mới giải thích

Format câu trả lời:
- Mở đầu bằng căn cứ pháp lý: "Theo Điều X [law_name]..."
- Giải thích chi tiết
- Kết luận/tóm tắt nếu cần"""


def build_context(articles):
    """Build context string from retrieved articles."""
    parts = []
    for art in articles:
        header = f"【{art.get('law_name', '')} - {art.get('article_num', '')}】"
        parts.append(f"{header}\n{art.get('text', '')}")
    return "\n\n".join(parts)


def build_user_prompt(question, context):
    """Build user prompt with question and context."""
    return f"""Dựa vào các điều luật sau để trả lời câu hỏi:

{context}

Câu hỏi: {question}

Hãy trả lời chi tiết, trích dẫn đúng điều luật, và nêu rõ căn cứ pháp lý."""


def _load_api_key(cfg):
    """Load Gemini API key from env var or .env file."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return ""


class GeminiGenerator:
    """LLM generator using Google Gemini API (google-genai SDK)."""

    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.model_name = self.cfg.get("llm", {}).get("gemini_model", "gemini-2.0-flash")
        self.temperature = self.cfg.get("llm", {}).get("temperature", 0.1)
        self.max_tokens = self.cfg.get("llm", {}).get("max_new_tokens", 2048)
        self.client = None

    def load(self):
        from google import genai

        api_key = _load_api_key(self.cfg)
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set env var GEMINI_API_KEY or create .env file."
            )

        self.client = genai.Client(api_key=api_key)
        return self

    def generate(self, question, articles):
        """Generate answer for a question given retrieved articles.

        Returns dict with answer, relevant_docs, relevant_articles.
        """
        context = build_context(articles)
        user_prompt = build_user_prompt(question, context)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        )
        answer = response.text

        # Extract cited articles from answer
        relevant_articles = extract_cited_articles(answer, articles)

        # Build relevant_docs (unique law_ids)
        relevant_docs = list({art["law_id"] for art in relevant_articles})

        return {
            "answer": answer,
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
        }


def extract_cited_articles(answer, articles):
    """Extract cited articles from answer text.

    Looks for patterns like 'Điều X' and matches against retrieved articles.
    """
    # Find all "Điều X" patterns in the answer
    dieu_pattern = re.compile(r"Điều\s+(\d+[a-z]?)", re.IGNORECASE)
    cited_nums = set(m.group(1) for m in dieu_pattern.finditer(answer))

    # Match against retrieved articles
    cited_articles = []
    seen = set()
    for art in articles:
        art_num = art.get("article_num", "")
        m = re.match(r"Điều\s+(\d+[a-z]?)", art_num, re.IGNORECASE)
        if m and m.group(1) in cited_nums:
            key = (art["law_id"], art["article_num"])
            if key not in seen:
                seen.add(key)
                cited_articles.append({
                    "law_id": art["law_id"],
                    "law_name": art.get("law_name", ""),
                    "article_num": art["article_num"],
                })

    # If no citations found, return all retrieved articles (for Recall)
    if not cited_articles:
        for art in articles:
            key = (art["law_id"], art["article_num"])
            if key not in seen:
                seen.add(key)
                cited_articles.append({
                    "law_id": art["law_id"],
                    "law_name": art.get("law_name", ""),
                    "article_num": art["article_num"],
                })

    return cited_articles


def main():
    cfg = load_config()
    generator = GeminiGenerator(cfg)
    generator.load()

    question = "Doanh nghiệp nhỏ và vừa được hưởng những hỗ trợ gì theo quy định của pháp luật?"
    test_articles = [
        {
            "law_id": "04/2017/QH14",
            "law_name": "Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
            "article_num": "Điều 12",
            "text": "Điều 12. Hỗ trợ về mặt bằng sản xuất, kinh doanh\nDoanh nghiệp nhỏ và vừa được ưu tiên thuê mặt bằng...",
        },
        {
            "law_id": "04/2017/QH14",
            "law_name": "Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
            "article_num": "Điều 13",
            "text": "Điều 13. Hỗ trợ về công nghệ\nDoanh nghiệp nhỏ và vừa được hỗ trợ chuyển giao công nghệ...",
        },
    ]

    result = generator.generate(question, test_articles)
    print(f"Answer:\n{result['answer']}\n")
    print(f"Relevant docs: {result['relevant_docs']}")
    print(f"Relevant articles: {result['relevant_articles']}")


if __name__ == "__main__":
    main()
