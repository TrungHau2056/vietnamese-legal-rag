#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[0-9a-zA-ZÀ-ỹĐđ]+", re.UNICODE)
ARTICLE_RE = re.compile(r"Điều\s+([0-9]+(?:\.[0-9]+)*)", re.IGNORECASE)
DOC_RE = re.compile(r"\b(Bộ luật|Luật|Nghị định|Thông tư liên tịch|Thông tư|Quyết định|Nghị quyết|Pháp lệnh|Lệnh|Chỉ thị)\s+(?:số\s+)?([0-9][^\s,;)]+)", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", str(text)).lower()
    return TOKEN_RE.findall(text)


def get_article_label(row: dict) -> str:
    text = f"{row.get('source_note_text') or ''} {row.get('article_title') or ''}"
    m = ARTICLE_RE.search(text)
    if m:
        return f"Điều {m.group(1)}"
    title = str(row.get("article_title") or "").strip()
    return title.split(".", 1)[0].strip() or "Điều"


def get_doc_info(row: dict) -> tuple[str, str]:
    source = str(row.get("source_note_text") or "")
    m = DOC_RE.search(source)
    code = m.group(2).strip() if m else str(row.get("doc_code") or row.get("law_id") or "UNKNOWN")
    name = source.strip() or str(row.get("doc_name") or row.get("title") or code)
    return code, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--top-k", type=int, default=4000)
    args = ap.parse_args()

    dataset = load_dataset("tmquan/phapdien-moj-gov-vn", "articles", split="train")
    rows = [dict(r) for r in dataset]
    contents = [str(r.get("text") or r.get("content") or r.get("article_content") or r.get("article_text") or r) for r in rows]
    bm25 = BM25Okapi([tokenize(c) for c in contents])

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8-sig"))
    output = []
    for q in questions:
        scores = bm25.get_scores(tokenize(q.get("question", "")))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: args.top_k]
        docs, arts = [], []
        seen_docs, seen_arts = set(), set()
        for i in top_idx:
            code, name = get_doc_info(rows[i])
            art_label = get_article_label(rows[i])
            doc = f"{code}|{name}"
            art = f"{code}|{name}|{art_label}"
            if doc not in seen_docs:
                docs.append(doc); seen_docs.add(doc)
            if art not in seen_arts:
                arts.append(art); seen_arts.add(art)
        output.append({"id": q.get("id"), "question": q.get("question", ""), "relevant_docs": docs, "relevant_articles": arts})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
