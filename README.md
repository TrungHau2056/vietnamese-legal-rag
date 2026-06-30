# R2AI 2026 - Legal Assistant Submission Package

## 1. Tổng quan

Repository này chứa mã nguồn, notebook, tài liệu thuyết minh, mô tả dữ liệu, thông tin mô hình và file kết quả cho bài dự thi R2AI 2026 - Build AI Legal Assistant.

Hệ thống được xây dựng theo hướng **truy hồi hai giai đoạn** và **không huấn luyện/fine-tune mô hình mới**:

1. **BM25 lexical retrieval**: lọc nhanh top-k1 điều luật/văn bản ứng viên dựa trên tín hiệu từ khóa.
2. **Dense embedding bi-encoder reranking/filtering**: dùng các checkpoint pretrained công khai trên Hugging Face để encode câu hỏi và điều luật, sau đó xếp hạng lại ứng viên.
3. **Output formatting**: xuất `results.json` theo định dạng yêu cầu gồm `id`, `question`, `answer`, `relevant_docs`, `relevant_articles`.

## 2. Khẳng định về huấn luyện mô hình

Đội **không train**, **không fine-tune**, **không tạo checkpoint riêng**. Toàn bộ bước dense retrieval/reranking sử dụng trực tiếp các mô hình embedding pretrained/public checkpoint:

| Model ID | Vai trò | Nguồn |
|---|---|---|
| `BAAI/bge-m3` | Embedding multilingual baseline, hỗ trợ dense retrieval | https://huggingface.co/BAAI/bge-m3 |
| `kietnt0603/nrk-legal-bge` | Embedding tiếng Việt pháp lý, dùng cho truy hồi luật | https://huggingface.co/kietnt0603/nrk-legal-bge |
| `AITeamVN/Vietnamese_Embedding` | Embedding tiếng Việt, dùng cho semantic reranking | https://huggingface.co/AITeamVN/Vietnamese_Embedding |

Lưu ý: một số checkpoint trên Hugging Face có thể đã được tác giả mô hình fine-tune trước khi công khai, nhưng đội thi chỉ **sử dụng lại pretrained/public weights**, không tự huấn luyện lại trên dữ liệu cuộc thi.

## 3. Cấu trúc thư mục

```text
├── README.md
├── requirements.txt
├── environment.yml
├── configs/
│   └── config.yaml
├── docs/
│   ├── R2AI_Tai_lieu_thuyet_minh_san_pham.docx
│   ├── data_description.md
│   ├── method.md
│   ├── model_checkpoint.md
│   ├── no_training_statement.md
├── notebooks/
│   ├── r2ai-bm25.ipynb
│   ├── r2ai-bi-encoder-threshold.ipynb
│   └── r2ai-embedding-top-k.ipynb
├── src/
│   ├── bm25_retriever.py
│   ├── dense_reranker.py
│   └── build_submission.py
├── scripts/
│   ├── run_pipeline.sh
│   ├── validate_submission.py
│   └── merge_questions_and_generate_answers.py
├── outputs/
│   ├── results.json
│   ├── submission.zip
│   ├── results_retrieval_only.json
│   ├── submission_retrieval_only.zip
│   ├── results_full_schema_TEMPLATE_NEED_REAL_QUESTIONS.json
│   └── submission_full_schema_TEMPLATE_NEED_REAL_QUESTIONS.zip
├── assets/
│   ├── scoring_result.zip
│   └── prediction_result.zip
```

## 4. Dữ liệu

Dataset chính: `tmquan/phapdien-moj-gov-vn`, config `articles`, split `train`.

Nguồn dữ liệu được dùng làm corpus pháp luật để truy hồi căn cứ pháp lý. Mỗi điều luật được chuẩn hóa thành một đơn vị văn bản để đánh chỉ mục BM25 và tạo embedding.

Xem chi tiết tại `docs/data_description.md`.

## 5. Mô hình và phương pháp

- Stage 1: BM25 với `rank_bm25.BM25Okapi`.
- Stage 2: dense embedding bi-encoder với **pretrained/public checkpoints**, không train/fine-tune.
- Model mặc định trong cấu hình hiện tại: `AITeamVN/Vietnamese_Embedding`.
- Các model có thể chọn: `BAAI/bge-m3`, `kietnt0603/nrk-legal-bge`, `AITeamVN/Vietnamese_Embedding`.
- Không sử dụng GPT-4o, Gemini hoặc mô hình đóng trong pipeline nộp bài.

Xem chi tiết tại `docs/method.md`, `docs/model_checkpoint.md`, `docs/no_training_statement.md`.

## 6. Cài đặt môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Trên Kaggle/Colab có thể chạy trực tiếp các notebook trong thư mục `notebooks/`.

## 7. Chạy pipeline

Cách chạy notebook gốc:

1. Chạy `notebooks/r2ai-bm25.ipynb` để tạo tập ứng viên BM25 top-k1.
2. Chạy `notebooks/r2ai-bi-encoder-threshold.ipynb` để rerank/filter bằng pretrained dense embedding.
3. Chạy `notebooks/r2ai-embedding-top-k.ipynb` nếu cần lọc theo score threshold hoặc top-k.
4. Kiểm tra định dạng kết quả bằng:

```bash
python scripts/validate_submission.py outputs/results.json
```

5. Tạo file zip nộp leaderboard:

```bash
cd outputs
zip -j submission.zip results.json
```

Có thể đổi model bằng biến môi trường:

```bash
EMBEDDING_MODEL_NAME="BAAI/bge-m3" bash scripts/run_pipeline.sh
EMBEDDING_MODEL_NAME="kietnt0603/nrk-legal-bge" bash scripts/run_pipeline.sh
EMBEDDING_MODEL_NAME="AITeamVN/Vietnamese_Embedding" bash scripts/run_pipeline.sh
```

## 8. Lưu ý quan trọng về file hiện tại

File `outputs/submission.zip` đã được tạo đúng cấu trúc zip: `results.json` nằm ngay ở thư mục gốc. Tuy nhiên, `results.json` hiện tại là bản **retrieval-only**, đang thiếu `question` và `answer` ở tất cả item theo kiểm tra tự động.

Do đó trước khi nộp chính thức theo schema đầy đủ, cần hợp nhất câu hỏi gốc và sinh/ghép câu trả lời pháp lý tiếng Việt cho từng câu hỏi.

Có thể dùng script hỗ trợ:

```bash
python scripts/merge_questions_and_generate_answers.py   --questions /path/to/test_questions.json   --retrieval outputs/results_retrieval_only.json   --output outputs/results.json
```

Sau đó chạy lại:

```bash
python scripts/validate_submission.py outputs/results.json
zip -j outputs/submission.zip outputs/results.json
```

## 9. Liên hệ / thông tin đội

- Tên đội: `uetay`
- Thành viên đại diện: `Châu Nguyễn Tố Trinh`
- Email/SĐT: `chaunguyentotrinh2006@gmail.com`
