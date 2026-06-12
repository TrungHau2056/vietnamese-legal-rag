# Processing Module

## Chunking Strategy

- **Primary**: Chunk at Điều (article) level — matches competition's article-level evaluation
- Điều dài → split at Khoản level, keep parent Điều reference
- Include parent context (Chương title) in chunk text for better embedding

## Markdown Parser (UTS_VLC)

- Parse markdown content → extract Điều/Khoản/Điểm hierarchy
- Pattern: `Điều X. <title>` → starts new article chunk
- Pattern: `Chương X - <title>` → metadata for chapter context

## phapdien Parser (tmquan/phapdien-moj-gov-vn)

- Already chunked by Điều — minimal processing needed
- Map columns: `article_title` → `article_num`, `content_text` → `text`, `chapter_title` → `chapter`
- Filter by `topic_title` for SME relevance

## law_name Format

Must follow: `Loại văn bản + Mã văn bản + Trích yếu`

Examples:
- `Luật 59/2020/QH14 Luật Doanh nghiệp`
- `Nghị định 01/2021/NĐ-CP về đăng ký doanh nghiệp`
- `Bộ luật 91/2015/QH13 Bộ luật Dân sự`
