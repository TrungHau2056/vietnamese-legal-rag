# Submission Module

## Output Format (results.json)

```json
[{
  "id": 1,
  "question": "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào...",
  "answer": "Theo Điều 4 Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa, ...",
  "relevant_docs": [
    "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
  ],
  "relevant_articles": [
    "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4"
  ]
}]
```

## Validation Rules

- `id`: integer, matches test set
- `question`: non-empty string
- `answer`: non-empty string, should contain "Điều X" patterns for auto-extraction
- `relevant_docs`: format `<mã_vb>|<tên_vb>`, tên = Loại VB + Mã VB + Trích yếu
- `relevant_articles`: format `<mã_vb>|<tên_vb>|<Điều X>`

## Submission

```bash
# Linux/macOS
zip submission.zip results.json
# Windows PowerShell
Compress-Archive -Path results.json -DestinationPath submission.zip
```

- File zip chỉ chứa `results.json` ở gốc
- Upload tại http://leaderboard.aiguru.com.vn/
- Max 10 submissions/ngày, 5 total in Private Phase
