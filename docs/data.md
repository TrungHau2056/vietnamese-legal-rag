# Data Documentation

## Datasets

### 1. `undertheseanlp/UTS_VLC` — Corpus toàn văn (markdown)

- **Splits**: 2026 (306), 2026_01 (318), 2023 (208), 2021 (110) = ~942 docs
- **Columns**: `id` (mã luật), `filename`, `title`, `type`, `content` (markdown toàn văn), `content_length`
- **Ví dụ**: id=91/2015/QH13, title="Bộ Luật dân sự", content=toàn văn markdown
- **Cần**: parse markdown → chunk theo Điều/Khoản

### 2. `tmquan/phapdien-moj-gov-vn` — Corpus đã chunk theo Điều (từ Bộ Tư pháp)

- **Columns**: `subject_id`, `topic_id`, `topic_number`, `topic_title`, `subject_number`, `subject_title`, `article_anchor`, `article_title`, `chapter_title`, `source_note_text`, `source_links`, `related_note_text`, `content_text`, `content_char_len`, `content_word_count`, `source_url`, `scraped_at`
- **Ưu điểm**: Đã chunk sẵn theo Điều, có `topic_title` để lọc SME, có `source_links` trỏ vbpl.vn
- **Lọc**: Dùng `topic_title` chứa keyword SME (Doanh nghiệp, Lao động, Thuế, Đầu tư...)

### 3. `adamwhite625/vietnam-legal-qa` — QA pairs

- **Columns**: `id`, `question`, `law_content`, `law_id`, `law_name`
- **Cách dùng**: Few-shot examples cho prompt, đánh giá retrieval quality

### 4. Bổ sung

- `uitnlp/ALQAC`, `Truong-Phan/ViLegal`, `nqminh0106/VietLaw`
- vbpl.vn (scrape), data.gov.vn (API)

## Schema: articles.jsonl (output sau khi processed)

```json
{
  "law_id": "59/2020/QH14",
  "law_name": "Luật 59/2020/QH14 Luật Doanh nghiệp",
  "law_type": "Luật",
  "article_num": "Điều 1",
  "chapter": "Chương I - NHỮNG QUY ĐỊNH CHUNG",
  "text": "Nội dung điều luật...",
  "source": "UTS_VLC"
}
```

## SME Document Filter List

**Core (bắt buộc)**: 59/2020/QH14, 04/2017/QH14, 45/2019/QH14, 38/2019/QH14, 91/2015/QH13, 61/2020/QH14
**Decrees**: 01/2021/NĐ-CP, 80/2021/NĐ-CP, 145/2020/NĐ-CP, 123/2020/NĐ-CP, 94/2023/NĐ-CP
**Extended**: Luật Thuế TNDN, Thuế GTGT, BHXH, Thương mại, Cạnh tranh
