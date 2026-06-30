# Mô hình sử dụng và checkpoint

## 1. Tuyên bố chính

Đội thi **không train**, **không fine-tune** và **không tạo checkpoint riêng**. Hệ thống sử dụng trực tiếp các pretrained/public embedding checkpoints từ Hugging Face để thực hiện dense retrieval/reranking ở chế độ inference-only.

Điều này có nghĩa là trong repository không có thư mục `training/`, không có script huấn luyện, không có log huấn luyện và không có trọng số mô hình do đội tự tạo. Các checkpoint được tải bằng Hugging Face Hub hoặc cache cục bộ.

## 2. Danh sách pretrained/public checkpoints

| Model ID | Loại mô hình | Vai trò trong pipeline | Nguồn tải |
|---|---|---|---|
| `BAAI/bge-m3` | Multilingual embedding model | Baseline semantic retrieval / dense embedding | https://huggingface.co/BAAI/bge-m3 |
| `kietnt0603/nrk-legal-bge` | Sentence-transformers model cho tiếng Việt pháp lý | Legal semantic reranking/filtering | https://huggingface.co/kietnt0603/nrk-legal-bge |
| `AITeamVN/Vietnamese_Embedding` | Vietnamese sentence embedding model | Vietnamese semantic reranking/filtering | https://huggingface.co/AITeamVN/Vietnamese_Embedding |

Model mặc định trong `configs/config.yaml` hiện là:

```yaml
reranking:
  model_name: AITeamVN/Vietnamese_Embedding
```

Có thể đổi model khi chạy bằng biến môi trường:

```bash
EMBEDDING_MODEL_NAME="BAAI/bge-m3" bash scripts/run_pipeline.sh
EMBEDDING_MODEL_NAME="kietnt0603/nrk-legal-bge" bash scripts/run_pipeline.sh
EMBEDDING_MODEL_NAME="AITeamVN/Vietnamese_Embedding" bash scripts/run_pipeline.sh
```

## 3. Vai trò của mô hình embedding

Các model embedding được dùng để:

- Encode câu hỏi pháp lý thành vector embedding.
- Encode nội dung điều luật ứng viên thành vector embedding.
- Tính điểm tương đồng giữa câu hỏi và điều luật.
- Rerank/filter các ứng viên đã được BM25 truy hồi.

Toàn bộ quá trình là inference-only, không cập nhật trọng số.

## 4. Cách tải và sử dụng checkpoint

Cài thư viện:

```bash
pip install FlagEmbedding sentence-transformers torch
```

Cách load với `sentence-transformers`:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")
embeddings = model.encode(["Câu hỏi pháp lý tiếng Việt"])
```

Cách load BGE-M3 với `FlagEmbedding`:

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
embeddings = model.encode(["Câu hỏi pháp lý tiếng Việt"])
```

Nếu chạy offline, tải checkpoint trước bằng Hugging Face CLI:

```bash
huggingface-cli download BAAI/bge-m3 --local-dir models/BAAI_bge-m3
huggingface-cli download kietnt0603/nrk-legal-bge --local-dir models/nrk-legal-bge
huggingface-cli download AITeamVN/Vietnamese_Embedding --local-dir models/Vietnamese_Embedding
```

Sau đó chỉnh `configs/config.yaml`:

```yaml
reranking:
  model_name: models/Vietnamese_Embedding
```

## 5. Tuân thủ quy định mô hình

- Không sử dụng GPT-4o, Gemini hoặc API/mô hình đóng trong pipeline nộp bài.
- Không huấn luyện mô hình mới trên dữ liệu cuộc thi.
- Không có checkpoint tự huấn luyện cần nộp.
- Các model sử dụng là public model IDs trên Hugging Face, có thể tải lại để tái lập pipeline.
- Nếu BTC yêu cầu kiểm tra thời điểm phát hành/checkpoint cụ thể, đội cần dẫn link Hugging Face model card/commit tương ứng trong email hoặc README.

## 6. Bảng khai báo checkpoint cho BTC

| Thành phần | Giá trị |
|---|---|
| Có train/fine-tune bởi đội không? | Không |
| Có checkpoint riêng không? | Không |
| Model mặc định | `AITeamVN/Vietnamese_Embedding` |
| Các model pretrained có thể dùng | `BAAI/bge-m3`, `kietnt0603/nrk-legal-bge`, `AITeamVN/Vietnamese_Embedding` |
| Nguồn tải checkpoint | Hugging Face model hub |
| Vị trí cache/local | Hugging Face cache hoặc `models/` nếu chạy offline |
| Mục đích | Dense embedding reranking/filtering sau BM25 |
