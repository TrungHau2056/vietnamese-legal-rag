# Nhật ký Tiến độ

## Trạng thái Đã xác minh Hiện tại

- Thư mục gốc kho lưu trữ: `d:\vietnamese-legal-rag`
- Đường dẫn khởi động chuẩn: `bash init.sh`
- Đường dẫn xác minh chuẩn: `.venv/Scripts/python.exe src/retrieval/hybrid_retriever.py` → test retrieval
- Tính năng chưa hoàn thành có mức ưu tiên cao nhất hiện tại: `gen-001` (LLM answer generation)
- Sự cố chặn hiện tại: Chưa có GEMINI_API_KEY — cần set env var hoặc tạo .env

## Nhật ký Phiên

### Phiên 003 — 2026-06-11

- Ngày: 2026-06-11
- Mục tiêu: Triển khai Phase 2 (indexing + retrieval)
- Đã hoàn thành:
  - Chạy BGE-M3 embedder → 2,666 vectors (1024 dim, max_length=512)
  - Build FAISS IndexFlatIP → indexes/faiss/index.faiss
  - Build BM25 index (pyvi tokenization) → indexes/bm25/bm25.pkl
  - Hybrid retrieval (dense + BM25 + RRF) → test 3 queries, kết quả chính xác
  - Cross-encoder reranker (BGE-reranker-v2-m3 via transformers) → test pass
  - End-to-end retrieval test: query → top-50 → rerank → top-10, kết quả đúng
  - Cập nhật generator.py dùng google-genai SDK mới
  - Tạo run_pipeline.py
- Xác minh đã chạy: idx-001 PASS, ret-001 PASS
- Bằng chứng: 3 test queries đều trả đúng luật + điều liên quan
- Rủi ro đã biết:
  - FlagEmbedding reranker không tương thích với transformers 5.x (đã fix bằng transformers trực tiếp)
  - Chưa có GEMINI_API_KEY cho Phase 3
  - BGE-M3 max_length=512 có thể miss context dài cho các Điều dài
- Bước tốt nhất tiếp theo: Set GEMINI_API_KEY → test generation → Phase 3

### Phiên 002 — 2026-06-10

- Ngày: 2026-06-10
- Mục tiêu: Chạy pipeline Phase 1 end-to-end, xác minh đầu ra
- Đã hoàn thành:
  - Chạy load_hf_datasets.py → tải 3 datasets thành công (942 UTS_VLC, 64K phapdien, 4.8K legal_qa)
  - Tạo .venv với Python 3.10
  - Sửa Python 3.13 torch conflict (chuyển sang Python 3.10)
  - Sửa UnicodeEncodeError trên Windows (PYTHONIOENCODING=utf-8)
  - Sửa UTS_VLC law_id mismatch:
    - Thêm `normalize_uts_law_id()` để extract standard law_id từ slug IDs
    - Thêm slug-to-law_id mapping cho known patterns
    - Cập nhật `build_law_name()` để dùng normalized law_id + Vietnamese titles
  - Mở rộng `extract_law_id_from_source_note()` cho TT-BCA, QĐ-TTg, TTLT-* patterns
  - Bỏ qua phapdien articles có empty law_id
  - Chạy chunker.py → 2,666 articles from 712 unique laws
  - End-to-end verification PASSED
- Xác minh đã chạy: Có — unit (schema) + system (pipeline flow) + end-to-end (6 core SME laws)
- Bằng chứng đã ghi lại: 2,666 articles, 6 core SME laws đều có articles, 0 duplicates, 0 empty law_id
- Commit: Chưa commit
- Rủi ro đã biết hoặc vấn đề chưa được giải quyết:
  - UTS_VLC slug mapping chỉ cover các law đã biết — nếu thêm luật mới cần cập nhật slug_map
  - Chưa có test set chính thức từ BTC
- Bước tốt nhất tiếp theo: Triển khai Phase 2 (indexing — build FAISS + BM25 indexes)

### Phiên 001 — 2026-06-10

- Ngày: 2026-06-10
- Mục tiêu: Thiết lập project, lên kế hoạch, tạo cấu trúc thư mục & docs
- Đã hoàn thành:
  - Phân tích yêu cầu cuộc thi R2AI2026
  - Nghiên cứu models (BGE-M3, Vistral-7B, Qwen2.5-7B)
  - Nghiên cứu datasets (UTS_VLC, phapdien, legal_qa)
  - Tạo PLAN.md, CLAUDE.md, docs/cho từng module
  - Tạo configs/config.yaml, requirements.txt
  - Viết src/data_collection/load_hf_datasets.py
  - Viết src/processing/chunker.py
- Xác minh đã chạy: Chưa — code đã viết nhưng chưa chạy pipeline
- Bước tốt nhất tiếp theo: Chạy load_hf_datasets.py → chunker.py → verify articles.jsonl
