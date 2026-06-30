# Tuyên bố không huấn luyện mô hình

Đội thi sử dụng hệ thống retrieval-based gồm BM25 và dense embedding bi-encoder. Đội **không thực hiện huấn luyện mô hình**, không fine-tune các embedding model và không tạo checkpoint riêng.

## Phạm vi sử dụng mô hình

Các model sau được sử dụng ở chế độ inference-only:

1. `BAAI/bge-m3`
2. `kietnt0603/nrk-legal-bge`
3. `AITeamVN/Vietnamese_Embedding`

Các model này được tải từ Hugging Face hoặc cache cục bộ. Pipeline chỉ dùng model để encode câu hỏi và điều luật, tính similarity, rerank/filter ứng viên và xuất căn cứ pháp lý.

## Những gì không có trong hệ thống

- Không có dữ liệu training riêng.
- Không có script training/fine-tuning.
- Không có checkpoint do đội tự huấn luyện.
- Không có thay đổi trọng số mô hình.
- Không sử dụng GPT-4o/Gemini hoặc mô hình đóng trong pipeline nộp bài.

## Cách tái lập

Người kiểm tra chỉ cần cài dependencies, tải dataset Pháp điển và tải model từ Hugging Face theo `docs/model_checkpoint.md`, sau đó chạy notebook/script inference để tái lập quá trình truy hồi.
