# Mô tả phương pháp

## 1. Tổng quan pipeline

Hệ thống sử dụng pipeline truy hồi hai giai đoạn để xác định căn cứ pháp lý liên quan cho từng câu hỏi. Đội **không train/fine-tune mô hình** mà dùng trực tiếp các checkpoint embedding pretrained/public trên Hugging Face.

```text
Question
  -> BM25 lexical retrieval, lấy top-k1 ứng viên
  -> Pretrained dense embedding bi-encoder reranking/filtering
  -> Chuẩn hóa relevant_docs / relevant_articles
  -> Ghép hoặc sinh answer dựa trên căn cứ truy hồi
  -> Xuất results.json
```

## 2. Stage 1 - BM25 lexical retrieval

BM25 được dùng để tìm nhanh các điều luật có độ khớp từ khóa cao với câu hỏi.

Các bước:

1. Tiền xử lý câu hỏi và nội dung điều luật.
2. Tokenize bằng regex hỗ trợ tiếng Việt, số và mã văn bản pháp luật.
3. Xây dựng chỉ mục bằng `rank_bm25.BM25Okapi`.
4. Với mỗi câu hỏi, tính điểm BM25 trên toàn bộ corpus.
5. Lấy top-k1 ứng viên. Trong notebook hiện tại, `TOP_K = 4000` cho bước BM25 để ưu tiên recall.

Ưu điểm của BM25:

- Nhanh, đơn giản, dễ tái lập.
- Bắt tốt các tín hiệu mã luật, số điều, cụm từ pháp lý chính xác.
- Phù hợp làm bước recall ứng viên ban đầu.

## 3. Stage 2 - Pretrained dense embedding bi-encoder reranking/filtering

Sau khi có tập ứng viên từ BM25, hệ thống dùng mô hình dense embedding dạng bi-encoder để biểu diễn câu hỏi và điều luật trong cùng không gian vector. Các mô hình được sử dụng ở chế độ inference-only, không cập nhật trọng số.

Các pretrained/public checkpoints được khai báo:

| Model ID | Mục đích sử dụng |
|---|---|
| `BAAI/bge-m3` | Embedding multilingual baseline, hỗ trợ truy hồi ngữ nghĩa. |
| `kietnt0603/nrk-legal-bge` | Embedding cho truy hồi pháp lý tiếng Việt. |
| `AITeamVN/Vietnamese_Embedding` | Embedding tiếng Việt, dùng để rerank/filter ứng viên. |

Các bước:

1. Load model embedding theo cấu hình `configs/config.yaml` hoặc biến môi trường `EMBEDDING_MODEL_NAME`.
2. Encode câu hỏi thành vector query.
3. Encode nội dung điều luật ứng viên thành vector passage.
4. Tính độ tương đồng query-passage bằng dot product/cosine similarity tùy model.
5. Rerank ứng viên và lọc theo `score_threshold` hoặc `top_k`.

Ưu điểm của dense bi-encoder:

- Bắt được tương đồng ngữ nghĩa tốt hơn BM25 khi câu hỏi diễn đạt khác từ khóa trong luật.
- Giảm số lượng điều luật nhiễu từ top-k1.
- Giúp chọn top-k2 hoặc nhóm ứng viên cuối cùng để đưa vào kết quả.

## 4. Không huấn luyện mô hình

Trong sản phẩm này, đội không thực hiện các bước sau:

- Không fine-tune `BAAI/bge-m3`.
- Không fine-tune `kietnt0603/nrk-legal-bge`.
- Không fine-tune `AITeamVN/Vietnamese_Embedding`.
- Không tạo custom checkpoint.
- Không sử dụng dữ liệu cuộc thi để huấn luyện lại mô hình embedding.
- Không sử dụng closed model/API như GPT-4o hoặc Gemini trong pipeline nộp bài.

Các notebook chỉ thực hiện inference: encode query, encode passage, tính similarity, rerank/filter và xuất kết quả.

## 5. Output formatting

Với mỗi câu hỏi, hệ thống xuất:

```json
{
  "id": 1,
  "question": "...",
  "answer": "...",
  "relevant_docs": [
    "<mã văn bản>|<tên văn bản>"
  ],
  "relevant_articles": [
    "<mã văn bản>|<tên văn bản>|<điều>"
  ]
}
```

## 6. Hạn chế hiện tại

File kết quả được upload hiện là bản retrieval-only, đã có `id`, `relevant_docs`, `relevant_articles`, nhưng còn thiếu `question` và `answer`. Cần bổ sung câu hỏi gốc từ file test và ghép/sinh câu trả lời pháp lý trước khi nộp chính thức theo schema đầy đủ.
