"""Parse raw legal documents and chunk by Điều (article) level.

Reads from data/raw/ → writes to data/processed/articles.jsonl
"""

import json
import os
import re
from collections import defaultdict

import yaml
from tqdm import tqdm


def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── UTS_VLC markdown parser ──────────────────────────────────────────────────

# Patterns for Vietnamese legal document structure
DIEU_PATTERN = re.compile(
    r"^\s*Điều\s+(\d+[a-z]?)\s*[\.\:]\s*(.*)", re.IGNORECASE
)
CHUONG_PATTERN = re.compile(
    r"^\s*Chương\s+([IVXLCDM]+|\d+)\s*[\.\-\s]+(.*)", re.IGNORECASE
)


def normalize_uts_law_id(raw_id):
    """Normalize UTS_VLC doc id to standard law_id format.

    UTS_VLC uses three ID styles:
    1. Standard: "59/2020/QH14", "04/2017/QH14"
    2. Slug with embedded code: "Luat-Dau-tu-so-61-2020-QH14-321051"
    3. Slug with year+name: "code-2019-bo-luat-lao-dong", "law-2020-luat-dau-tu"
    4. Slug with name only: "bo-luat-dan-su", "luat-dau-tu"
    """
    # Already standard format (XX/YYYY/QHXX or XX/YYYY/NĐ-CP etc.)
    if re.match(r"^\d{2}/\d{4}/", raw_id):
        return raw_id

    # Try to extract standard code from slug (e.g. "61-2020-QH14" → "61/2020/QH14")
    m = re.search(r"(\d{1,3})-(\d{4})-(QH\d+)", raw_id)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"

    # Slug-to-law_id mapping for known patterns
    slug_map = {
        # Core SME laws — slug variants
        "bo-luat-dan-su": "91/2015/QH13",
        "Bo-luat-dan-su-2015-296215": "91/2015/QH13",
        "code-2015-bo-luat-dan-su": "91/2015/QH13",
        "bo-luat-lao-dong": "45/2019/QH14",
        "Bo-Luat-lao-dong-2019-333670": "45/2019/QH14",
        "code-2019-bo-luat-lao-dong": "45/2019/QH14",
        "luat-dau-tu": "61/2020/QH14",
        "law-2020-luat-dau-tu": "61/2020/QH14",
        "luat-doanh-nghiep": "59/2020/QH14",
        "law-2020-luat-doanh-nghiep": "59/2020/QH14",
        "Luat-Doanh-nghiep-so-59-2020-QH14-427301": "59/2020/QH14",
        "law-2017-luat-ho-tro-doanh-nghiep-nho-va-vua": "04/2017/QH14",
        "Luat-Ho-tro-doanh-nghiep-nho-va-vua-2017-320905": "04/2017/QH14",
        "law-2019-luat-quan-ly-thue": "38/2019/QH14",
        "Luat-quan-ly-thue-2019-387595": "38/2019/QH14",
        # Extended laws — slug variants
        "Luat-Bao-hiem-xa-hoi-2014-259700": "71/2015/QH13",
        "law-2014-luat-bao-hiem-xa-hoi": "71/2015/QH13",
        "luat-bao-hiem-xa-hoi": "71/2015/QH13",
        "Luat-Thuong-mai-2005-36-2005-QH11-2633": "36/2005/QH11",
        "law-2005-luat-thuong-mai": "36/2005/QH11",
        "luat-thuong-mai": "36/2005/QH11",
        "Luat-canh-tranh-345182": "23/2018/QH14",
        "law-2018-luat-canh-tranh": "23/2018/QH14",
        "luat-canh-tranh": "23/2018/QH14",
    }

    if raw_id in slug_map:
        return slug_map[raw_id]

    # No normalization found — return as-is
    return raw_id


# Vietnamese law title → standard law_id mapping for build_law_name
LAW_TITLE_TO_ID = {
    "Bộ Luật dân sự": "91/2015/QH13",
    "Bo Luat Dan Su": "91/2015/QH13",
    "Bo luat dan su 2015 296215": "91/2015/QH13",
    "Bộ luật Lao động": "45/2019/QH14",
    "Bo Luat Lao Dong": "45/2019/QH14",
    "Bo Luat lao dong 2019 333670": "45/2019/QH14",
    "Luật Đầu tư": "61/2020/QH14",
    "Luat Dau Tu": "61/2020/QH14",
    "Luat Dau tu so 61 2020 QH14 321051": "61/2020/QH14",
    "Luật Doanh nghiệp": "59/2020/QH14",
    "Luat Doanh Nghiep": "59/2020/QH14",
    "Luat Doanh nghiep so 59 2020 QH14 427301": "59/2020/QH14",
    "Luật Hỗ trợ doanh nghiệp nhỏ và vừa": "04/2017/QH14",
    "Luat Ho tro doanh nghiep nho va vua 2017 320905": "04/2017/QH14",
    "Luật Quản lý thuế": "38/2019/QH14",
    "Luat quan ly thue 2019 387595": "38/2019/QH14",
}


def extract_law_id_from_uts(doc):
    """Extract and normalize law_id from UTS_VLC doc id field."""
    raw_id = doc.get("id", "")
    return normalize_uts_law_id(raw_id)


def build_law_name(doc, normalized_law_id):
    """Build law_name in format: Loại văn bản + Mã văn bản + Trích yếu."""
    title = doc.get("title", "")
    doc_type = doc.get("type", "")

    type_map = {
        "code": "Bộ luật",
        "law": "Luật",
        "ordinance": "Pháp lệnh",
        "decree": "Nghị định",
        "circular": "Thông tư",
        "decision": "Quyết định",
        "resolution": "Nghị quyết",
    }
    loai_vb = type_map.get(doc_type, "Luật")

    # Use LAW_TITLE_TO_ID in reverse to get proper Vietnamese title
    id_to_title = {v: k for k, v in LAW_TITLE_TO_ID.items()}
    # Pick the Vietnamese (non-ASCII) title as trích yếu
    trich_yeu = ""
    for t, lid in LAW_TITLE_TO_ID.items():
        if lid == normalized_law_id and any(ord(c) > 127 for c in t):
            trich_yeu = t
            break
    if not trich_yeu:
        # Fall back to the doc title if it has Vietnamese chars
        if any(ord(c) > 127 for c in title):
            trich_yeu = title
        else:
            trich_yeu = ""

    law_name = f"{loai_vb} {normalized_law_id} {trich_yeu}".strip() if trich_yeu else f"{loai_vb} {normalized_law_id}"
    return law_name, loai_vb


def parse_uts_vlc_markdown(content, law_id, law_name, loai_vb):
    """Parse UTS_VLC markdown content into Điều-level chunks."""
    lines = content.split("\n")
    articles = []

    current_chapter = ""
    current_dieu_num = None
    current_dieu_title = ""
    current_dieu_lines = []

    def flush_article():
        if current_dieu_num is None:
            return
        text = "\n".join(current_dieu_lines).strip()
        if not text:
            return
        articles.append({
            "law_id": law_id,
            "law_name": law_name,
            "law_type": loai_vb,
            "article_num": f"Điều {current_dieu_num}",
            "chapter": current_chapter,
            "text": text,
            "source": "UTS_VLC",
        })

    for line in lines:
        # Check for Chương header
        chuong_match = CHUONG_PATTERN.match(line)
        if chuong_match:
            current_chapter = line.strip()
            continue

        # Check for Điều header
        dieu_match = DIEU_PATTERN.match(line)
        if dieu_match:
            # Flush previous article
            flush_article()

            current_dieu_num = dieu_match.group(1)
            current_dieu_title = dieu_match.group(2).strip()
            # First line of article text includes the title
            current_dieu_lines = [f"Điều {current_dieu_num}. {current_dieu_title}"]
            continue

        # Accumulate text under current Điều
        if current_dieu_num is not None:
            current_dieu_lines.append(line)

    # Flush last article
    flush_article()

    return articles


def process_uts_vlc(cfg):
    """Process UTS_VLC raw data into Điều-level chunks."""
    raw_dir = os.path.join(cfg["paths"]["raw_data"], "uts_vlc")
    if not os.path.exists(raw_dir):
        print("  [skip] UTS_VLC raw data not found. Run load_hf_datasets.py first.")
        return []

    all_articles = []

    for filename in sorted(os.listdir(raw_dir)):
        if not filename.endswith(".jsonl"):
            continue
        filepath = os.path.join(raw_dir, filename)
        print(f"  Processing {filename}...")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"  Parsing {filename}"):
                doc = json.loads(line)
                law_id = extract_law_id_from_uts(doc)
                law_name, loai_vb = build_law_name(doc, law_id)

                content = doc.get("content", "")
                if not content:
                    continue

                articles = parse_uts_vlc_markdown(content, law_id, law_name, loai_vb)
                all_articles.extend(articles)

    return all_articles


# ── phapdien parser ───────────────────────────────────────────────────────────

# Law ID patterns for extraction from source_note_text
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


def extract_law_id_from_source_note(source_note):
    """Extract law_id from phapdien source_note_text."""
    if not source_note:
        return ""
    for pat in _LAW_ID_PATTERNS:
        m = pat.search(source_note)
        if m:
            return m.group(1)
    return ""


def build_law_name_from_source(source_note, law_id):
    """Build law_name in format: Loại VB + Mã VB + Trích yếu."""
    if not source_note:
        return law_id

    # Extract document type and trích yếu from source note
    # Example: "(Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia ngày 03/12/2004 ...)"
    loai_order = ["Bộ luật", "Luật", "Pháp lệnh", "Nghị định", "Thông tư", "Quyết định", "Nghị quyết"]

    loai_vb = ""
    for vn_type in loai_order:
        if vn_type in source_note:
            loai_vb = vn_type
            break

    if not loai_vb:
        return law_id

    # Extract trích yếu: text between law_id and "ngày"/"của"
    if law_id:
        idx = source_note.find(law_id)
        if idx != -1:
            remainder = source_note[idx + len(law_id):].strip()
            # Cut at common delimiters
            for stop in ["ngày", "của Quốc hội", "của Chính phủ", "của Bộ", "của Ngân hàng",
                         "của Thủ tướng", "của Ủy ban", ", có hiệu lực", "  "]:
                ri = remainder.find(stop)
                if ri > 0:
                    remainder = remainder[:ri].strip()
            trich_yeu = remainder.strip(" ,.")
            if trich_yeu:
                return f"{loai_vb} {law_id} {trich_yeu}"

    return f"{loai_vb} {law_id}"


def process_phapdien(cfg):
    """Process phapdien raw data — take ALL articles, merge by (law_id, article_num).

    phapdien splits each Điều into Khoản/Điểm rows. We merge them back into
    Điều-level chunks to match UTS_VLC granularity and evaluation criteria.
    """
    raw_dir = os.path.join(cfg["paths"]["raw_data"], "phapdien")
    if not os.path.exists(raw_dir):
        print("  [skip] phapdien raw data not found. Run load_hf_datasets.py first.")
        return []

    # First pass: collect all rows grouped by (law_id, article_num)
    groups = defaultdict(list)
    law_names = {}
    law_types = {}
    chapters = {}
    skipped_no_law_id = 0
    skipped_empty = 0
    total_rows = 0

    for filename in sorted(os.listdir(raw_dir)):
        if not filename.endswith(".jsonl"):
            continue
        filepath = os.path.join(raw_dir, filename)
        print(f"  Processing {filename}...")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"  Reading {filename}"):
                doc = json.loads(line)
                total_rows += 1

                source_note = doc.get("source_note_text", "")
                law_id = extract_law_id_from_source_note(source_note)

                if not law_id:
                    skipped_no_law_id += 1
                    continue

                content_text = doc.get("content_text", "")
                if not content_text or len(content_text.strip()) < 10:
                    skipped_empty += 1
                    continue

                article_title = doc.get("article_title", "")
                dieu_match = re.match(r"Điều\s+(\d+)", article_title)
                if not dieu_match:
                    continue
                article_num = f"Điều {dieu_match.group(1)}"

                key = (law_id, article_num)
                groups[key].append(content_text)

                # Store law_name (prefer first occurrence which is usually the fullest)
                if key not in law_names:
                    law_names[key] = build_law_name_from_source(source_note, law_id)
                    # Determine law_type from loai_vb prefix
                    name = law_names[key]
                    if name.startswith("Bộ luật"):
                        law_types[key] = "Bộ luật"
                    elif name.startswith("Luật"):
                        law_types[key] = "Luật"
                    elif name.startswith("Nghị định"):
                        law_types[key] = "Nghị định"
                    elif name.startswith("Thông tư"):
                        law_types[key] = "Thông tư"
                    elif name.startswith("Quyết định"):
                        law_types[key] = "Quyết định"
                    else:
                        law_types[key] = ""
                    chapters[key] = doc.get("chapter_title", "")

    print(f"  Total rows: {total_rows}")
    print(f"  Skipped (no law_id): {skipped_no_law_id}")
    print(f"  Skipped (empty text): {skipped_empty}")
    print(f"  Unique (law_id, article_num) groups: {len(groups)}")

    # Second pass: merge text within each group into a single Điều
    all_articles = []
    for (law_id, article_num), texts in groups.items():
        # Deduplicate identical texts (phapdien may have duplicates)
        seen_texts = list(dict.fromkeys(texts))
        # Merge: join with newline, prepend article_num as header
        merged_text = "\n".join(seen_texts)

        all_articles.append({
            "law_id": law_id,
            "law_name": law_names.get((law_id, article_num), law_id),
            "law_type": law_types.get((law_id, article_num), ""),
            "article_num": article_num,
            "chapter": chapters.get((law_id, article_num), ""),
            "text": merged_text,
            "source": "phapdien",
        })

    return all_articles


# ── Merge & deduplicate ──────────────────────────────────────────────────────

def deduplicate_articles(articles):
    """Deduplicate by (law_id, article_num) pair. Prefer UTS_VLC over phapdien."""
    seen = {}
    for art in articles:
        key = (art["law_id"], art["article_num"])
        if key not in seen:
            seen[key] = art
        else:
            # Prefer UTS_VLC (parsed from official markdown) over phapdien
            existing = seen[key]
            if existing["source"] != "UTS_VLC" and art["source"] == "UTS_VLC":
                seen[key] = art
    return list(seen.values())


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    os.makedirs(cfg["paths"]["processed_data"], exist_ok=True)

    print("=== Phase 1: Data Collection & Processing ===\n")

    # Step 1: Process UTS_VLC
    print("1. Processing UTS_VLC (markdown → Điều-level chunks)")
    uts_articles = process_uts_vlc(cfg)
    print(f"   UTS_VLC: {len(uts_articles)} articles\n")

    # Step 2: Process phapdien
    print("2. Processing phapdien (filter SME → format)")
    phapdien_articles = process_phapdien(cfg)
    print(f"   phapdien: {len(phapdien_articles)} articles\n")

    # Step 3: Merge & deduplicate
    print("3. Merging & deduplicating")
    all_articles = uts_articles + phapdien_articles
    print(f"   Before dedup: {len(all_articles)} articles")
    deduped = deduplicate_articles(all_articles)
    print(f"   After dedup: {len(deduped)} articles\n")

    # Step 4: Save
    out_path = os.path.join(cfg["paths"]["processed_data"], "articles.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for art in deduped:
            f.write(json.dumps(art, ensure_ascii=False) + "\n")
    print(f"4. Saved to {out_path}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total articles: {len(deduped)}")
    law_counts = defaultdict(int)
    for art in deduped:
        law_counts[art["law_id"]] += 1
    print(f"Unique laws: {len(law_counts)}")
    for law_id, count in sorted(law_counts.items()):
        print(f"  {law_id}: {count} articles")


if __name__ == "__main__":
    main()
