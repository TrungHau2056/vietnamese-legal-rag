# Generation Module

## LLM Options

| Model | Context | Pros | Cons |
|-------|---------|------|------|
| `VietAI/Vistral-7B-Chat` | 8K | Tiếng Việt tự nhiên nhất | Context ngắn, cần retrieval tốt |
| `Qwen/Qwen2.5-7B-Instruct` | 128K | Context dài, reasoning mạnh | Tiếng Việt vừa phải, có thể code-switch |

## Prompt Design

- **System**: Trợ lý pháp lý, phải trích dẫn điều luật, cảnh báo giới hạn AI
- **Format yêu cầu**: "Theo Điều X [Loại] Mã_văn_bản Tên_văn_bản, ..."
- **Few-shot**: Dùng samples từ `adamwhite625/vietnam-legal-qa`

## Citation Extraction

- Parse answer text cho pattern "Điều X"
- Match với retrieved articles theo law_id + article_num
- Validate citations tồn tại trong corpus
- Build `relevant_docs` + `relevant_articles` từ citations + retrieved top articles

## Inference

- Use `vllm` hoặc `transformers` với 4-bit quantization (bitsandbytes)
- 24GB VRAM → đủ chạy 7B model + embedding/reranker xen kẽ
