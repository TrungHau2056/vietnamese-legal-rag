"""Load HuggingFace datasets and save raw data to disk."""

import json
import os
import sys

import yaml
from datasets import load_dataset
from tqdm import tqdm


def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_uts_vlc(cfg):
    """Load undertheseanlp/UTS_VLC — full-law corpus in markdown."""
    name = cfg["datasets"]["uts_vlc"]["name"]
    splits = cfg["datasets"]["uts_vlc"]["splits"]
    out_dir = os.path.join(cfg["paths"]["raw_data"], "uts_vlc")
    os.makedirs(out_dir, exist_ok=True)

    all_docs = []
    for split in splits:
        out_path = os.path.join(out_dir, f"{split}.jsonl")
        if os.path.exists(out_path):
            print(f"  [skip] {split} already saved at {out_path}")
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    all_docs.append(json.loads(line))
            continue

        print(f"  Loading {name} split={split}...")
        ds = load_dataset(name, split=split)
        with open(out_path, "w", encoding="utf-8") as f:
            for item in tqdm(ds, desc=f"  Saving {split}"):
                doc = {
                    "id": item["id"],
                    "filename": item.get("filename", ""),
                    "title": item.get("title", ""),
                    "type": item.get("type", ""),
                    "content": item.get("content", ""),
                    "content_length": item.get("content_length", 0),
                    "source": "UTS_VLC",
                    "split": split,
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                all_docs.append(doc)

    print(f"  UTS_VLC: {len(all_docs)} documents total")
    return all_docs


def load_phapdien(cfg):
    """Load tmquan/phapdien-moj-gov-vn — chunked by Điều from Bộ Tư pháp."""
    name = cfg["datasets"]["phapdien"]["name"]
    split = cfg["datasets"]["phapdien"]["split"]
    out_dir = os.path.join(cfg["paths"]["raw_data"], "phapdien")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{split}.jsonl")
    if os.path.exists(out_path):
        print(f"  [skip] phapdien already saved at {out_path}")
        with open(out_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    print(f"  Loading {name} split={split}...")
    ds = load_dataset(name, split=split)
    all_items = []
    with open(out_path, "w", encoding="utf-8") as f:
        for item in tqdm(ds, desc="  Saving phapdien"):
            doc = {
                "topic_id": item.get("topic_id", ""),
                "topic_number": item.get("topic_number", 0),
                "topic_title": item.get("topic_title", ""),
                "subject_number": item.get("subject_number", 0),
                "subject_title": item.get("subject_title", ""),
                "article_anchor": item.get("article_anchor", ""),
                "article_title": item.get("article_title", ""),
                "chapter_title": item.get("chapter_title", ""),
                "source_note_text": item.get("source_note_text", ""),
                "content_text": item.get("content_text", ""),
                "content_char_len": item.get("content_char_len", 0),
                "content_word_count": item.get("content_word_count", 0),
                "source_url": item.get("source_url", ""),
                "source": "phapdien",
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            all_items.append(doc)

    print(f"  phapdien: {len(all_items)} articles total")
    return all_items


def load_legal_qa(cfg):
    """Load adamwhite625/vietnam-legal-qa — QA pairs."""
    name = cfg["datasets"]["legal_qa"]["name"]
    split = cfg["datasets"]["legal_qa"]["split"]
    out_dir = os.path.join(cfg["paths"]["raw_data"], "legal_qa")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{split}.jsonl")
    if os.path.exists(out_path):
        print(f"  [skip] legal_qa already saved at {out_path}")
        with open(out_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    print(f"  Loading {name} split={split}...")
    ds = load_dataset(name, split=split)
    all_items = []
    with open(out_path, "w", encoding="utf-8") as f:
        for item in tqdm(ds, desc="  Saving legal_qa"):
            doc = {
                "id": item.get("id", ""),
                "question": item.get("question", ""),
                "law_content": item.get("law_content", ""),
                "law_id": item.get("law_id", ""),
                "law_name": item.get("law_name", ""),
                "source": "legal_qa",
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            all_items.append(doc)

    print(f"  legal_qa: {len(all_items)} QA pairs total")
    return all_items


def main():
    cfg = load_config()
    os.makedirs(cfg["paths"]["raw_data"], exist_ok=True)

    print("=== Loading HuggingFace datasets ===")

    print("\n1. UTS_VLC (full-law corpus)")
    uts_docs = load_uts_vlc(cfg)

    print("\n2. phapdien-moj-gov-vn (chunked by Điều)")
    phapdien_docs = load_phapdien(cfg)

    print("\n3. adamwhite625/vietnam-legal-qa (QA pairs)")
    qa_docs = load_legal_qa(cfg)

    print(f"\n=== Done ===")
    print(f"UTS_VLC: {len(uts_docs)} docs")
    print(f"phapdien: {len(phapdien_docs)} articles")
    print(f"legal_qa: {len(qa_docs)} QA pairs")
    print(f"Raw data saved to: {cfg['paths']['raw_data']}")


if __name__ == "__main__":
    main()
