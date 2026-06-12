# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Vietnamese Legal RAG cho cuộc thi **R2AI2026 BUILD AI LEGAL ASSISTANT** (deadline: 30/06/2026).
Hệ thống truy hồi điều luật (IR) + hỏi đáp pháp lý (QA) cho SME, tiếng Việt.

## Constraints

- Models <14B params, open-source, released trước 01/03/2026, no closed-source LLMs
- Output: `results.json` — `id`, `question`, `answer`, `relevant_docs`, `relevant_articles`
- `law_name` = Loại văn bản + Mã văn bản + Trích yếu (VD: "Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa")
- Submission: zip chứa `results.json` ở gốc → http://leaderboard.aiguru.com.vn/

## Evaluation

- **IR**: F2 = (5*P*R)/(4*P+R) — Recall weighted 2x → ưu tiên Recall, retrieve broad, rerank aggressive
- **QA**: 5 criteria — căn cứ pháp luật (auto), chính xác, đầy đủ, thực tiễn, rõ ràng (LLM-Judge + human)
- Auto-extract "Điều X" từ `answer` field để chấm IR

## Architecture

### Pipeline Flow (tuần tự, không skip bước)

```
data_collection → processing → indexing → retrieval → generation → submission
      ↓               ↓            ↓           ↓            ↓            ↓
  raw docs      articles.jsonl  FAISS+BM25  top-10 arts   answer+cite  results.json
```

Mỗi module nhận output của module trước, không gọi ngược. Output mỗi module là file trên disk (JSONL/index), không truyền in-memory giữa module.

### Component Models

| Component | Model |
|-----------|-------|
| Embedding | `BAAI/bge-m3` (dense+sparse+ColBERT, 8192 ctx) |
| Reranker | `BAAI/bge-reranker-v2-m3` (8192 ctx) |
| LLM | `VietAI/Vistral-7B-Chat` (8K) / `Qwen/Qwen2.5-7B-Instruct` (128K) |
| BM25 | `rank_bm25` |

### Architecture Rules

- **Config-driven**: Mọi hyperparam (top-k, RRF k, model name, paths) phải đọc từ `configs/config.yaml`, không hardcode
- **No external data in pipeline**: Không dùng data ngoài trong bất kỳ bước xử lý nào (quy tắc cuộc thi). Dữ liệu corpus phải thu thập trước và lưu trong `data/`
- **Disk-based I/O giữa modules**: Mỗi module ghi output ra file, module sau đọc file → dễ debug, retry, và chạy riêng lẻ
- **law_name format bắt buộc**: `Loại văn bản + Mã văn bản + Trích yếu` — sai format = không được chấm

## Route Map

Khi cần chi tiết, đọc docs trong folder tương ứng:

| Cần gì | Đọc ở đâu |
|--------|-----------|
| Kế hoạch triển khai tổng thể | [`PLAN.md`](PLAN.md) |
| Thông tin datasets, data schema | [`docs/data.md`](docs/data.md) |
| Chi tiết chunking, parsing strategy | [`src/processing/README.md`](src/processing/README.md) |
| Embedding & indexing config | [`src/indexing/README.md`](src/indexing/README.md) |
| Retrieval & reranking params | [`src/retrieval/README.md`](src/retrieval/README.md) |
| Prompt templates & LLM config | [`src/generation/README.md`](src/generation/README.md) |
| Pipeline orchestration | [`src/pipeline/README.md`](src/pipeline/README.md) |
| Output format & validation rules | [`src/submission/README.md`](src/submission/README.md) |
| Hyperparameters & config | [`configs/config.yaml`](configs/config.yaml) |
| Test set & sample data | [`data/test/`](data/test/) |

## Key Decisions

- Chunk at Điều level → matches article-level evaluation
- Hybrid retrieval: semantic + BM25 → BM25 catches exact law IDs
- BGE-M3 hybrid mode (dense+sparse) > pure dense for legal text
- Citation extraction từ answer text + retrieved articles
- SME corpus: ~50-100 văn bản (không chỉ luật doanh nghiệp — cần Dân sự, Lao động, Thuế, Đầu tư...)

## Verification Rule

Một tính năng chỉ được coi là **hoàn thành** khi qua cả 3 tầng kiểm tra:

1. **Unit test** — Hàm/class riêng lẻ hoạt động đúng (mock data, edge cases)
2. **System test** — Tích hợp giữa các module trong pipeline (dùng sample data thực)
3. **End-to-end** — Chạy pipeline đầy đủ trên test set → output `results.json` đúng format, truy vấn mẫu trả về articles chính xác

Nếu fail ở tầng nào → fix rồi kiểm lại từ tầng đó trở đi. Không skip tầng.

Mỗi lỗi phải báo đủ 3 thứ:
1. **What** — Cái gì sai (hàm nào, module nào, test case nào fail)
2. **Why** — Tại sao sai (nguyên nhân gốc: logic sai, edge case, data format...)
3. **How to fix** — Cách sửa (code change cụ thể, không chỉ gợi ý chung)

## Data Sources

- **HuggingFace**: `undertheseanlp/UTS_VLC` (corpus), `tmquan/phapdien-moj-gov-vn` (chunked by Điều), `adamwhite625/vietnam-legal-qa` (QA pairs)
- **Official**: vbpl.vn, data.gov.vn
- Vietnamese tokenization: `underthesea` / `pyvi`
