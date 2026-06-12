# Tài liệu Chất lượng

**Tần suất cập nhật:** Sau mỗi phiên quan trọng, hoặc trước khi bắt đầu giai đoạn công việc mới.

**Thang điểm:**
- **A**: Tất cả xác minh đang vượt qua, kiến trúc sạch, agent có thể đọc được, test ổn định
- **B**: Xác minh đang vượt qua, hầu hết sạch, thiếu nhỏ về khả năng đọc hoặc độ phủ test
- **C**: Hoạt động một phần, có khoảng trống đã biết, một số khu vực mã khó cho agent hiểu
- **D**: Không hoạt động, hoặc có vấn đề cấu trúc lớn

---

## Domain Sản phẩm

| Domain | Điểm | Xác minh | Khả năng đọc Agent | Độ ổn định Test | Khoảng trống chính | Cập nhật lần cuối |
|--------|-------|-------------|-----------------|---------------|----------|-------------|
| Thu thập Dữ liệu | A | PASS — 3 datasets loaded | A | A | Không | 2026-06-10 |
| Xử lý & Chunking | A | PASS — 2,666 articles, E2E verified | A | A | slug_map cần mở rộng khi thêm luật mới | 2026-06-10 |
| Indexing | A | PASS — FAISS + BM25 built | A | A | max_length=512 có thể miss context dài | 2026-06-11 |
| Retrieval | A | PASS — hybrid + rerank E2E verified | A | A | Không | 2026-06-11 |
| Generation | - | - | - | - | Chưa có GEMINI_API_KEY | - |

## Lớp Kiến trúc

| Lớp | Điểm | Thực thi Ranh giới | Khả năng đọc Agent | Khoảng trống chính | Cập nhật lần cuối |
|-------|-------|---------------------|-----------------|----------|-------------|
| Data Collection | A | Config-driven ✓, .venv ✓ | Tốt | Không | 2026-06-10 |
| Processing | A | Disk I/O ✓, law_id normalization ✓ | Tốt | slug_map cần mở rộng | 2026-06-10 |
| Indexing | A | Config-driven ✓, Disk I/O ✓ | Tốt | max_length=512 cho CPU | 2026-06-11 |
| Retrieval | A | RRF fusion ✓, reranker via transformers ✓ | Tốt | Không | 2026-06-11 |
| Generation | - | - | - | Chưa triển khai | - |
| Submission | - | - | - | Chưa triển khai | - |

## Lịch sử Thay đổi

### 2026-06-10 (Phiên 002)

- Thay đổi: Chạy pipeline Phase 1 end-to-end, sửa UTS_VLC law_id mismatch
- Domain được nâng cấp: Thu thập Dữ liệu (D→A), Xử lý & Chunking (D→A)
- Domain bị hạ cấp: N/A
- Khoảng trống mới được xác định: UTS_VLC slug_map chỉ cover known laws
- Khoảng trống đã đóng: UTS_VLC law_id mismatch, phapdien empty law_id, Python 3.13 torch conflict, UnicodeEncodeError

### 2026-06-10 (Phiên 001)

- Thay đổi: Khởi tạo project, tạo cấu trúc, viết scripts Phase 1
- Domain được nâng cấp: N/A (mới tạo)
- Domain bị hạ cấp: N/A
- Khoảng trống mới được xác định: HuggingFace SSL timeout, Python 3.13 torch conflict
- Khoảng trống đã đóng: N/A
