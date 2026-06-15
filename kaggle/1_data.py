"""Stage 1: Data Collection & Processing

INPUT:  Nothing (downloads from HuggingFace)
OUTPUT: articles.jsonl (~52K articles)

Chạy trên Kaggle:
    !pip install -q datasets pyvi tqdm
    !python 1_data.py

Fix chính:
- UTS_VLC law_id normalization: dùng title-based lookup từ split 2026
  để resolve slug IDs (code-2015-bo-luat-dan-su → 91/2015/QH13)
- Skip docs không resolve được (slug law_id = 0 điểm khi chấm)
- Cap article text tại 8192 chars (loại bỏ merge artifacts dài 8M chars)
- Fix regex \d{2} → \d{1,3} cho law_id 3 chữ số
"""

import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from tqdm import tqdm

from datasets import load_dataset

# ── Config ──────────────────────────────────────────────────────────────
ARTICLES_PATH = "articles.jsonl"
MAX_ARTICLE_CHARS = 8192  # cap để tránh garbage embeddings

UTS_SPLITS = ["2026", "2026_01", "2023", "2021"]

DIEU_RE = re.compile(r"^\s*Điều\s+(\d+[a-z]?)\s*[\.\:]\s*(.*)", re.IGNORECASE)
CHUONG_RE = re.compile(r"^\s*Chương\s+([IVXLCDM]+|\d+)\s*[\.\-\s]+(.*)", re.IGNORECASE)

# ── Law ID extraction (phapdien) ────────────────────────────────────────
_LAW_ID_PATTERNS = [
    re.compile(r"(\d{1,3}/\d{4}/QH\d+)"),
    re.compile(r"(\d{1,3}/\d{4}/NĐ-CP)"),
    re.compile(r"(\d{1,3}/\d{4}/ND-CP)"),
    re.compile(r"(\d{1,3}/\d{4}/TT-[A-ZĐ]+)"),
    re.compile(r"(\d{1,3}/\d{4}/TTLT-[A-ZĐ]+)"),
    re.compile(r"(\d{1,3}/\d{4}/QĐ-[A-ZĐ]+)"),
    re.compile(r"(\d{1,3}/\d{4}/QD-[A-ZĐ]+)"),
    re.compile(r"(\d{1,3}/\d{4}/PL-UBTVQH\d+)"),
    re.compile(r"(\d{1,3}/\d{4}/NQ-[A-ZĐ]+)"),
]


def extract_law_id_phapdien(source_note):
    if not source_note:
        return ""
    for pat in _LAW_ID_PATTERNS:
        m = pat.search(source_note)
        if m:
            return m.group(1)
    return ""


def build_law_name_phapdien(source_note, law_id):
    if not source_note:
        return law_id
    loai_order = ["Bộ luật", "Luật", "Pháp lệnh", "Nghị định", "Thông tư", "Quyết định", "Nghị quyết"]
    loai_vb = ""
    for vn_type in loai_order:
        if vn_type in source_note:
            loai_vb = vn_type
            break
    if not loai_vb:
        return law_id
    if law_id:
        idx = source_note.find(law_id)
        if idx != -1:
            remainder = source_note[idx + len(law_id) :].strip()
            for stop in [
                "ngày", "của Quốc hội", "của Chính phủ", "của Bộ", "của Ngân hàng",
                "của Thủ tướng", "của Ủy ban", ", có hiệu lực", "  ",
            ]:
                ri = remainder.find(stop)
                if ri > 0:
                    remainder = remainder[:ri].strip()
            trich_yeu = remainder.strip(" ,.")
            if trich_yeu:
                return f"{loai_vb} {law_id} {trich_yeu}"
    return f"{loai_vb} {law_id}"


# ── Law ID normalization (UTS_VLC) ──────────────────────────────────────

def _normalize_title(text):
    """Chuyển title về dạng không dấu, lowercase để so sánh."""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def build_title_lookup(uts_split="2026"):
    """Build lookup: normalized_title → law_id từ split 2026 (có ID chuẩn).

    Đây là nguồn truth — split 2026 dùng standard law_ids.
    """
    print(f"Building title lookup from UTS_VLC split={uts_split}...")
    lookup = {}
    ds = load_dataset("undertheseanlp/UTS_VLC", split=uts_split)
    for doc in tqdm(ds, desc=f"Building lookup from {uts_split}"):
        raw_id = doc.get("id", "")
        title = doc.get("title", "")
        # Chỉ dùng docs có standard law_id làm lookup source
        if re.match(r"^\d{1,3}/\d{4}/", raw_id):
            norm_title = _normalize_title(title)
            if norm_title and norm_title not in lookup:
                lookup[norm_title] = raw_id
    print(f"  Lookup size: {len(lookup)} entries")
    return lookup


def build_phapdien_lookup():
    """Build lookup: normalized_law_name → law_id từ phapdien source_note_text."""
    print("Building phapdien lookup...")
    lookup = {}
    ds = load_dataset("tmquan/phapdien-moj-gov-vn", split="train")
    seen = set()
    for item in tqdm(ds, desc="Scanning phapdien"):
        source_note = item.get("source_note_text", "")
        law_id = extract_law_id_phapdien(source_note)
        if not law_id or law_id in seen:
            continue
        seen.add(law_id)
        # Extract loai + name từ source_note
        loai_order = ["Bộ luật", "Luật", "Pháp lệnh", "Nghị định", "Thông tư", "Quyết định", "Nghị quyết"]
        loai_vb = ""
        for vn_type in loai_order:
            if vn_type in source_note:
                loai_vb = vn_type
                break
        if loai_vb and law_id:
            idx = source_note.find(law_id)
            if idx != -1:
                remainder = source_note[idx + len(law_id) :].strip()
                for stop in ["ngày", "của Quốc hội", "của Chính phủ", "của Bộ", "của Ngân hàng",
                             "của Thủ tướng", "của Ủy ban", ", có hiệu lực", "  "]:
                    ri = remainder.find(stop)
                    if ri > 0:
                        remainder = remainder[:ri].strip()
                trich_yeu = remainder.strip(" ,.")
                if trich_yeu:
                    full_name = f"{loai_vb} {trich_yeu}"
                    norm = _normalize_title(full_name)
                    if norm and norm not in lookup:
                        lookup[norm] = law_id
    print(f"  Phapdien lookup size: {len(lookup)} entries")
    return lookup


def normalize_uts_law_id(raw_id, title, title_lookup, phapdien_lookup):
    """Multi-strategy UTS_VLC law_id resolver.

    Returns law_id (string) nếu resolve được, None nếu không.
    """
    # 1. Standard format: XX/YYYY/...
    if re.match(r"^\d{1,3}/\d{4}/", raw_id):
        return raw_id

    # 2. Embedded code trong slug: Luat-Dau-tu-so-61-2020-QH14-321051
    m = re.search(r"(\d{1,3})-(\d{4})-(QH\d+)", raw_id)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"

    # 3. Title-based lookup (so sánh title không dấu)
    if title:
        norm_title = _normalize_title(title)
        if norm_title in title_lookup:
            return title_lookup[norm_title]

        # Thử bỏ suffix năm+ID (format 2023: "Bo Luat to tung dan su 2015")
        # Strip trailing year patterns
        stripped = re.sub(r"\s+\d{4}\s*$", "", norm_title)
        if stripped != norm_title and stripped in title_lookup:
            return title_lookup[stripped]

    # 4. Phapdien lookup (law name → law_id)
    if title:
        norm_title = _normalize_title(title)
        if norm_title in phapdien_lookup:
            return phapdien_lookup[norm_title]

    # 5. Không resolve được → trả None (sẽ skip doc này)
    return None


def build_law_name_uts(doc, law_id):
    doc_type = doc.get("type", "")
    title = doc.get("title", "")
    type_map = {
        "code": "Bộ luật", "law": "Luật", "ordinance": "Pháp lệnh",
        "decree": "Nghị định", "circular": "Thông tư",
        "decision": "Quyết định", "resolution": "Nghị quyết",
    }
    loai_vb = type_map.get(doc_type, "Luật")

    # Ưu tiên title tiếng Việt có dấu làm trích yếu
    trich_yeu = ""
    if any(ord(c) > 127 for c in title):
        trich_yeu = title
    return f"{loai_vb} {law_id} {trich_yeu}".strip() if trich_yeu else f"{loai_vb} {law_id}"


def parse_uts_markdown(content, law_id, law_name):
    lines = content.split("\n")
    articles = []
    current_chapter = ""
    current_dieu_num = None
    current_dieu_lines = []

    def flush():
        if current_dieu_num is None:
            return
        text = "\n".join(current_dieu_lines).strip()
        if text:
            if len(text) > MAX_ARTICLE_CHARS:
                text = text[:MAX_ARTICLE_CHARS]
            articles.append({
                "law_id": law_id,
                "law_name": law_name,
                "article_num": f"Điều {current_dieu_num}",
                "chapter": current_chapter,
                "text": text,
                "source": "UTS_VLC",
            })

    for line in lines:
        chuong_match = CHUONG_RE.match(line)
        if chuong_match:
            current_chapter = line.strip()
            continue
        dieu_match = DIEU_RE.match(line)
        if dieu_match:
            flush()
            current_dieu_num = dieu_match.group(1)
            current_dieu_lines = [f"Điều {current_dieu_num}. {dieu_match.group(2).strip()}"]
            continue
        if current_dieu_num is not None:
            current_dieu_lines.append(line)
    flush()
    return articles


# ── Main pipeline ───────────────────────────────────────────────────────

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("STAGE 1: DATA COLLECTION & PROCESSING")
    print("=" * 60)

    # ── Build lookups cho law_id normalization ──
    title_lookup = build_title_lookup("2026")
    phapdien_lookup = build_phapdien_lookup()

    # ── 1a. Process phapdien ──
    print("\n--- 1a. phapdien ---")
    ds = load_dataset("tmquan/phapdien-moj-gov-vn", split="train")
    print(f"  Raw rows: {len(ds)}")

    phapdien_groups = defaultdict(list)
    phapdien_law_names = {}
    phapdien_chapters = {}
    skipped = 0

    for item in tqdm(ds, desc="Processing phapdien"):
        source_note = item.get("source_note_text", "")
        law_id = extract_law_id_phapdien(source_note)
        if not law_id:
            skipped += 1
            continue
        content = item.get("content_text", "")
        if not content or len(content.strip()) < 10:
            skipped += 1
            continue
        article_title = item.get("article_title", "")
        m = re.match(r"Điều\s+(\d+)", article_title)
        if not m:
            continue
        article_num = f"Điều {m.group(1)}"
        key = (law_id, article_num)
        phapdien_groups[key].append(content)
        if key not in phapdien_law_names:
            phapdien_law_names[key] = build_law_name_phapdien(source_note, law_id)
            phapdien_chapters[key] = item.get("chapter_title", "")

    print(f"  Skipped: {skipped}")
    print(f"  Unique (law_id, article_num): {len(phapdien_groups)}")

    phapdien_articles = []
    for (law_id, article_num), texts in phapdien_groups.items():
        seen_texts = list(dict.fromkeys(texts))
        merged_text = "\n".join(seen_texts)
        if len(merged_text) > MAX_ARTICLE_CHARS:
            merged_text = merged_text[:MAX_ARTICLE_CHARS]
        phapdien_articles.append({
            "law_id": law_id,
            "law_name": phapdien_law_names.get((law_id, article_num), law_id),
            "article_num": article_num,
            "chapter": phapdien_chapters.get((law_id, article_num), ""),
            "text": merged_text,
            "source": "phapdien",
        })
    print(f"  phapdien articles: {len(phapdien_articles)}")

    # ── 1b. Process UTS_VLC ──
    print("\n--- 1b. UTS_VLC ---")
    uts_articles = []
    skipped_no_law_id = 0

    for split in UTS_SPLITS:
        print(f"\n  Loading UTS_VLC split={split}...")
        try:
            ds = load_dataset("undertheseanlp/UTS_VLC", split=split)
        except Exception as e:
            print(f"    Split {split} failed: {e}. Skipping.")
            continue
        print(f"    {len(ds)} docs in {split}")

        split_resolved = 0
        split_skipped = 0

        for doc in tqdm(ds, desc=f"  Parsing {split}"):
            raw_id = doc.get("id", "")
            title = doc.get("title", "")

            law_id = normalize_uts_law_id(raw_id, title, title_lookup, phapdien_lookup)
            if law_id is None:
                split_skipped += 1
                continue

            law_name = build_law_name_uts(doc, law_id)
            content = doc.get("content", "")
            if not content:
                continue

            parsed = parse_uts_markdown(content, law_id, law_name)
            uts_articles.extend(parsed)
            split_resolved += 1

        print(f"    Resolved: {split_resolved}, Skipped (no law_id): {split_skipped}")
        print(f"    Running total: {len(uts_articles)} articles")
        skipped_no_law_id += split_skipped

    print(f"\n  UTS_VLC total: {len(uts_articles)} articles")
    print(f"  UTS_VLC skipped (unresolved law_id): {skipped_no_law_id} docs")

    # ── 1c. Merge + deduplicate ──
    print("\n--- 1c. Merge & Dedup ---")
    all_articles = uts_articles + phapdien_articles
    print(f"  Before dedup: {len(all_articles)} (UTS_VLC: {len(uts_articles)}, phapdien: {len(phapdien_articles)})")

    # Dedup by (law_id, article_num) — prefer UTS_VLC
    seen = {}
    for art in all_articles:
        key = (art["law_id"], art["article_num"])
        if key not in seen:
            seen[key] = art
        else:
            existing = seen[key]
            if existing["source"] != "UTS_VLC" and art["source"] == "UTS_VLC":
                seen[key] = art

    articles = list(seen.values())
    print(f"  After dedup: {len(articles)} articles")

    # ── 1d. Save ──
    with open(ARTICLES_PATH, "w", encoding="utf-8") as f:
        for art in articles:
            f.write(json.dumps(art, ensure_ascii=False) + "\n")
    print(f"\n  Saved to {ARTICLES_PATH}")

    # ── 1e. Summary ──
    law_counts = defaultdict(int)
    source_counts = defaultdict(int)
    for art in articles:
        law_counts[art["law_id"]] += 1
        source_counts[art["source"]] += 1

    print(f"\n  Total articles: {len(articles)}")
    print(f"  By source: {dict(source_counts)}")
    print(f"  Unique laws: {len(law_counts)}")

    # Check: có slug law_id nào còn sót không?
    slug_count = sum(1 for lid in law_counts if not re.match(r"^\d{1,3}/\d{4}/", lid))
    if slug_count > 0:
        print(f"\n  WARNING: {slug_count} laws still have slug-format law_ids!")
        for lid, count in sorted(law_counts.items(), key=lambda x: -x[1])[:20]:
            if not re.match(r"^\d{1,3}/\d{4}/", lid):
                print(f"    SLUG: {lid} ({count} articles)")
    else:
        print("\n  All law_ids are in standard format!")

    print("\n  Top 10 laws by article count:")
    for law_id, count in sorted(law_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {law_id}: {count} articles")

    print(f"\n{'=' * 60}")
    print(f"STAGE 1 COMPLETE: {ARTICLES_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
