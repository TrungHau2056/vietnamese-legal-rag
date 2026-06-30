#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

REQUIRED = ["id", "question", "answer", "relevant_docs", "relevant_articles"]


def validate_json(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    errors = []
    warnings = []

    if not isinstance(data, list):
        errors.append("Root must be a list.")
        data = []

    ids = set()
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"Item {idx} is not an object.")
            continue
        for key in REQUIRED:
            if key not in item:
                errors.append(f"Item index={idx}, id={item.get('id')} missing required field: {key}")
        if "id" in item:
            if item["id"] in ids:
                warnings.append(f"Duplicated id: {item['id']}")
            ids.add(item["id"])
        if "question" in item and not isinstance(item["question"], str):
            errors.append(f"Item id={item.get('id')} question must be string.")
        if "answer" in item and not isinstance(item["answer"], str):
            errors.append(f"Item id={item.get('id')} answer must be string.")
        for key in ["relevant_docs", "relevant_articles"]:
            if key in item and not isinstance(item[key], list):
                errors.append(f"Item id={item.get('id')} {key} must be list.")
        for doc in item.get("relevant_docs", []) if isinstance(item.get("relevant_docs"), list) else []:
            if not isinstance(doc, str) or doc.count("|") < 1:
                errors.append(f"Item id={item.get('id')} invalid relevant_doc: {doc!r}")
        for art in item.get("relevant_articles", []) if isinstance(item.get("relevant_articles"), list) else []:
            if not isinstance(art, str) or art.count("|") < 2:
                errors.append(f"Item id={item.get('id')} invalid relevant_article: {art!r}")

    print(f"Checked: {path}")
    print(f"Number of items: {len(data)}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    for e in errors[:50]:
        print("ERROR:", e)
    if len(errors) > 50:
        print(f"... {len(errors) - 50} more errors")
    for w in warnings[:20]:
        print("WARNING:", w)

    return 1 if errors else 0


def validate_zip(path: Path) -> int:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    print(f"Zip entries: {names}")
    if names != ["results.json"]:
        print("ERROR: zip must contain only results.json at root.")
        return 1
    print("Zip structure OK.")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: validate_submission.py <results.json|submission.zip>")
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        return 2
    if path.suffix.lower() == ".zip":
        return validate_zip(path)
    return validate_json(path)


if __name__ == "__main__":
    raise SystemExit(main())
