# Bàn giao Phiên

## Đang Hoạt động Hiện tại

- Những gì đang hoạt động: Phase 1 (data collection + processing) hoàn thành, end-to-end verification PASSED
- Xác minh nào thực sự đã chạy: data-001 PASS, data-002 PASS (2,666 articles, 6 core SME laws, 0 duplicates)

## Thay đổi Trong Phiên này

- Mã hoặc hành vi đã thêm:
  - Sửa `src/processing/chunker.py`:
    - Thêm `normalize_uts_law_id()` — chuyển UTS_VLC slug IDs thành standard law_ids
    - Thêm slug-to-law_id mapping (16 entries cho core + extended SME laws)
    - Thêm `LAW_TITLE_TO_ID` mapping cho Vietnamese title → law_id
    - Cập nhật `build_law_name()` — dùng normalized law_id + Vietnamese titles
    - Mở rộng `extract_law_id_from_source_note()` — thêm patterns cho TT-BCA, QĐ-TTg, TTLT-*, QD-TT
    - Bỏ qua phapdien articles có empty law_id
  - Tạo `.venv` với Python 3.10
  - Chạy `load_hf_datasets.py` → 3 datasets loaded
  - Chạy `chunker.py` → 2,666 articles in data/processed/articles.jsonl
- Thay đổi cơ sở hạ tầng:
  - `.venv` setup (Python 3.10, tất cả dependencies installed)
  - `PYTHONIOENCODING=utf-8` cho Windows Vietnamese output

## Bị Hỏng hoặc Chưa được Xác minh

- Lỗi đã biết: Không có lỗi hiện tại
- Đường dẫn chưa được xác minh: indexing → retrieval → generation → submission (Phase 2-5)
- Rủi ro cho phiên tiếp theo: UTS_VLC slug_map chỉ cover known laws — nếu thêm luật mới cần mở rộng

## Bước Tốt nhất Tiếp theo

- Tính năng chưa hoàn thành có mức ưu tiên cao nhất: `idx-001` (Build FAISS + BM25 indexes)
- Tại sao đây là tính năng tiếp theo: Có data rồi → cần index để retrieval
- Điều gì được tính là vượt qua: indexes/faiss/ + indexes/bm25/ tồn tại, query sample trả về kết quả
- Điều gì không được thay đổi trong bước đó: data/processed/, configs/, chunker logic

## Lệnh

- Khởi động: `bash init.sh`
- Chạy Python: `.venv/Scripts/python.exe`
- Tải datasets: `.venv/Scripts/python.exe src/data_collection/load_hf_datasets.py`
- Xử lý & chunk: `.venv/Scripts/python.exe src/processing/chunker.py`
- Lệnh debug: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "import json; ..."`
