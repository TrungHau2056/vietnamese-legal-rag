#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_rule_based_answer(item: dict) -> str:
    """Create a safe placeholder answer from retrieved citations.

    This is only a fallback. For final scoring, replace with a legal answer generated
    from the retrieved article contents and the original question.
    """
    articles = item.get("relevant_articles") or []
    docs = item.get("relevant_docs") or []

    if articles:
        parts = []
        for art in articles[:5]:
            fields = str(art).split("|")
            if len(fields) >= 3:
                parts.append(f"{fields[2]} của {fields[1]}")
            else:
                parts.append(str(art))
        return (
            "Theo các căn cứ pháp lý được truy hồi gồm "
            + "; ".join(parts)
            + ", cần đối chiếu nội dung câu hỏi với các quy định nêu trên để xác định quyền, nghĩa vụ hoặc chế tài tương ứng. "
            + "Câu trả lời cuối cùng nên được rà soát và bổ sung nội dung pháp lý cụ thể trước khi nộp chính thức."
        )
    if docs:
        return "Căn cứ các văn bản pháp luật liên quan: " + "; ".join(docs[:5]) + "."
    return "Chưa truy hồi được căn cứ pháp lý phù hợp để trả lời câu hỏi."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, help="JSON list with fields id and question")
    parser.add_argument("--retrieval", required=True, help="Retrieval-only results JSON")
    parser.add_argument("--output", required=True, help="Output full-schema results.json")
    parser.add_argument("--keep-existing-answer", action="store_true")
    args = parser.parse_args()

    questions = read_json(Path(args.questions))
    retrieval = read_json(Path(args.retrieval))

    qmap = {item["id"]: item["question"] for item in questions if "id" in item and "question" in item}

    output = []
    for item in retrieval:
        q = qmap.get(item.get("id"), item.get("question", ""))
        ans = item.get("answer") if args.keep_existing_answer and item.get("answer") else build_rule_based_answer(item)
        output.append({
            "id": item.get("id"),
            "question": q,
            "answer": ans,
            "relevant_docs": item.get("relevant_docs", []),
            "relevant_articles": item.get("relevant_articles", []),
        })

    write_json(Path(args.output), output)
    print(f"Wrote {args.output} with {len(output)} items")


if __name__ == "__main__":
    main()
