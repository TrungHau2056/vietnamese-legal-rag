# Tài liệu mô tả dữ liệu

## 1. Nguồn dữ liệu

Hệ thống sử dụng bộ dữ liệu pháp luật `tmquan/phapdien-moj-gov-vn` trên Hugging Face, config `articles`, split `train`. Dataset được dùng làm corpus truy hồi cho bài toán hỏi đáp pháp lý tiếng Việt.

## 2. Vai trò trong hệ thống

Dữ liệu Pháp điển được sử dụng để:

- Xây dựng chỉ mục BM25 phục vụ truy hồi từ khóa.
- Tạo tập điều luật ứng viên top-k1 cho mỗi câu hỏi.
- Làm nguồn văn bản đầu vào cho bước dense embedding bi-encoder.
- Sinh danh sách `relevant_docs` và `relevant_articles` trong file kết quả.

## 3. Cấu trúc dữ liệu sử dụng

Ở mức xử lý nội bộ, mỗi điều luật được chuẩn hóa thành một đối tượng gồm:

- `content`: nội dung điều luật hoặc phần văn bản pháp luật liên quan.
- `doc_code`: mã văn bản pháp luật, ví dụ `04/2017/QH14`.
- `doc_name`: tên văn bản pháp luật đầy đủ.
- `article_label`: nhãn điều luật, ví dụ `Điều 4`.
- `relevant_doc`: chuỗi định dạng `<mã văn bản>|<tên văn bản>`.
- `relevant_article`: chuỗi định dạng `<mã văn bản>|<tên văn bản>|<điều>`.

## 4. Tiền xử lý

Các bước tiền xử lý chính:

1. Đọc dataset từ Hugging Face bằng `datasets.load_dataset`.
2. Chuẩn hóa Unicode, lowercase và tách token tiếng Việt/chuỗi số/ký hiệu văn bản.
3. Trích xuất mã văn bản pháp luật bằng regex từ `source_note_text`.
4. Trích xuất nhãn điều luật theo mẫu `Điều X`.
5. Tạo chuỗi `relevant_docs` và `relevant_articles` theo format BTC yêu cầu.
6. Loại bỏ hoặc xử lý các trường rỗng nếu không đủ thông tin văn bản/điều luật.

## 5. Hướng dẫn tải dữ liệu

Cài thư viện:

```bash
pip install datasets
```

Tải dữ liệu:

```python
from datasets import load_dataset

dataset = load_dataset(
    "tmquan/phapdien-moj-gov-vn",
    "articles",
    split="train",
)
```

Nếu cần chạy offline, tải dataset về Google Drive hoặc thư mục `data/raw/`, sau đó chỉnh đường dẫn trong `configs/config.yaml`.

## 6. Lưu ý về dữ liệu ngoài

Đội sử dụng thêm dữ liệu ngoài:

| Dữ liệu | Nguồn | Cách thu thập | Vai trò | Link truy cập |
|---|---|---|---|---|
| Pháp điển | Hugging Face `tmquan/phapdien-moj-gov-vn` | `datasets.load_dataset` | Corpus truy hồi | `https://huggingface.co/datasets/tmquan/phapdien-moj-gov-vn` |
| Test questions | BTC cung cấp | Tải từ hệ thống cuộc thi | Đầu vào dự đoán |  |
