# Rubric Evaluator — Vietnamese Legal RAG

Sử dụng rubric này sau khi triển khai và trước khi chấp nhận cuối cùng.

| Hạng mục | Câu hỏi | Điểm (0-2) | Ghi chú |
| --- | --- | --- | --- |
| Tính đúng đắn | Output có đúng format cuộc thi? (id, question, answer, relevant_docs, relevant_articles) |  |  |
| Xác minh | Unit → System → End-to-end đều chạy? Có by chứng? |  |  |
| Kỷ luật phạm vi | Chỉ sửa trong phạm vi tính năng, không sửa module khác? |  |  |
| law_name format | `Loại VB + Mã VB + Trích yếu` đúng chưa? (VD: "Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa") |  |  |
| F2 Recall | Retrieval có ưu tiên Recall không? (retrieve broad, rerank aggressive) |  |  |
| Citation | Answer có chứa "Điều X" patterns? relevant_articles populated đúng? |  |  |
| Model constraints | Model <14B, open-source, released trước 01/03/2026? |  |  |
| Khả năng bảo trì | Config-driven, không hardcode? Disk-based I/O giữa modules? |  |  |
| Sẵn sàng bàn giao | `results.json` pass validation? Zip đúng cấu trúc? |  |  |

## Kết luận

- Chấp nhận
- Sửa đổi
- Chặn

## Hành động Tiếp theo Bắt buộc

- Bằng chứng còn thiếu:
- Sửa chữa bắt buộc:
- Kích hoạt review tiếp theo:
