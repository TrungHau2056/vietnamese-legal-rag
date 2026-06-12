# Danh sách Kiểm tra Trạng thái Sạch

- [x] `bash init.sh` chạy không lỗi
- [x] `data/processed/articles.jsonl` tồn tại và có nội dung
- [x] Mỗi article trong articles.jsonl có đủ fields (law_id, law_name, law_type, article_num, chapter, text, source)
- [x] Ít nhất 6 core SME laws có articles (59/2020=218, 04/2017=35, 45/2019=220, 38/2019=152, 91/2015=689, 61/2020=77)
- [x] Không có duplicate (law_id, article_num) pairs
- [x] law_name format đúng: Loại VB + Mã VB + Trích yếu
- [x] Không có empty law_id
- [x] Tiến độ hiện tại được ghi lại trong claude-progress.md
- [x] Trạng thái tính năng phản ánh những gì thực sự đang vượt qua (feature_list.json: data-001, data-002 = passing)
- [x] Không có bước làm dở nào bị bỏ lại mà không được ghi lại
- [x] Phiên tiếp theo có thể tiếp tục mà không cần sửa chữa thủ công
