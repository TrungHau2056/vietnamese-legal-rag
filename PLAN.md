# Vietnamese Legal RAG - Kế hoạch triển khai

## Bối cảnh

Xây dựng hệ thống Truy hồi & Hỏi đáp Văn bản Pháp luật Tiếng Việt cho cuộc thi **R2AI2026 BUILD AI LEGAL ASSISTANT**.

- **Deadline**: 30/06/2026 (còn ~21 ngày)
- **Hardware**: RTX 3090/4090 (24GB VRAM)
- **Ràng buộc**: Model <14B params, open-source, phát hành trước 01/03/2026
- **Đánh giá IR**: F2 (ưu tiên Recall 2x Precision)
- **Đánh giá QA**: 5 tiêu chí (căn cứ pháp luật, chính xác, đầy đủ, thực tiễn, rõ ràng)

---

## Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────┐
│                  Dữ liệu đầu vào                      │
│  vbpl.vn (scrape) + HuggingFace datasets              │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│            Document Parser & Chunker                   │
│  Parse HTML → Chunk theo Điều/Khoản/Điểm              │
│  Metadata: law_id, law_name, article_num, chapter     │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              BGE-M3 Encoder (568M)                     │
│  Dense + Sparse + ColBERT embeddings                   │
└──────────┬──────────────────────────┬───────────────┘
           ▼                          ▼
┌────────────────────┐   ┌────────────────────────────┐
│   FAISS Index       │   │     BM25 Index              │
│   (dense vectors)   │   │  (tokenized Vietnamese)     │
└────────┬───────────┘   └──────────┬─────────────────┘
         └──────────┬───────────────┘
                    ▼
┌─────────────────────────────────────────────────────┐
│          Hybrid Retrieval (RRF Fusion)                 │
│  Dense top-50 + BM25 top-50 → RRF merge → top-50      │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│        BGE-Reranker-v2-m3 (Cross-Encoder)             │
│  top-50 → Rerank → top-10                              │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│          LLM Generation (Vistral-7B / Qwen2.5-7B)     │
│  Prompt: question + retrieved context                  │
│  Output: answer + citations (Điều X Luật Y...)         │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│            Post-processing & Formatting                │
│  Extract citations → validate → results.json           │
└─────────────────────────────────────────────────────┘
```

---

## Model Stack

| Thành phần | Model | Tham số | Context | Lý do chọn |
|------------|-------|---------|---------|------------|
| **Embedding** | `BAAI/bge-m3` | 568M | 8192 | Hybrid dense+sparse+ColBERT, hỗ trợ tiếng Việt tốt |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | 568M | 8192 | Đồng bộ với bge-m3, cross-encoder chính xác cao |
| **LLM A** | `VietAI/Vistral-7B-Chat` | 7B | 8K | Tiếng Việt tự nhiên nhất |
| **LLM B** | `Qwen/Qwen2.5-7B-Instruct` | 7.62B | 128K | Context dài, reasoning mạnh |
| **BM25** | `rank_bm25` | - | - | Bắt chính xác mã luật, số điều |

---

## Cấu trúc thư mục

```
vietnamese-legal-rag/
├── data/
│   ├── raw/                    # Văn bản gốc (HTML/PDF)
│   ├── processed/
│   │   └── articles.jsonl      # Đã chunk theo Điều, có metadata
│   └── test/
│       └── test_questions.json
├── indexes/
│   ├── faiss/                  # FAISS dense index
│   └── bm25/                   # BM25 sparse index
├── src/
│   ├── data_collection/
│   │   ├── scrape_vbpl.py      # Crawl vbpl.vn
│   │   ├── load_hf_datasets.py # Load HuggingFace datasets
│   │   └── merge_corpora.py    # Gộp & loại trùng
│   ├── processing/
│   │   ├── parser.py           # Parse HTML → text có cấu trúc
│   │   ├── chunker.py          # Chunk theo Điều/Khoản/Điểm
│   │   └── metadata.py         # Trích xuất law_id, số điều, v.v.
│   ├── indexing/
│   │   ├── embedder.py         # BGE-M3 embedding
│   │   ├── faiss_index.py      # Build & query FAISS
│   │   └── bm25_index.py       # Build & query BM25
│   ├── retrieval/
│   │   ├── hybrid_retriever.py # Gộp dense + BM25 (RRF)
│   │   └── reranker.py         # BGE-reranker reranking
│   ├── generation/
│   │   ├── llm_loader.py       # Load Vistral-7B / Qwen2.5-7B
│   │   ├── prompt_templates.py # Prompt templates cho legal QA
│   │   └── generator.py        # Sinh câu trả lời + trích dẫn
│   ├── pipeline/
│   │   └── run_pipeline.py     # Pipeline end-to-end
│   └── submission/
│       ├── format_output.py    # Format theo yêu cầu cuộc thi
│       └── validate.py         # Validate results.json
├── configs/
│   └── config.yaml
├── requirements.txt
└── PLAN.md
```

---

## Giai đoạn triển khai

### Giai đoạn 1: Thu thập & xử lý dữ liệu (Ngày 1-3)

**1.1. Load HuggingFace datasets (3 nguồn chính)**

| Dataset | Loại | Columns | Số dòng | Cách dùng |
|---------|------|---------|---------|-----------|
| `undertheseanlp/UTS_VLC` | Corpus toàn văn (markdown) | `id`, `filename`, `title`, `type`, `content`, `content_length` | 306+318+208+110 = ~942 docs | **Corpus chính** — toàn văn luật, cần chunk theo Điều |
| `tmquan/phapdien-moj-gov-vn` | Corpus từ Bộ Tư pháp (đã chunk theo Điều) | `subject_id`, `topic_id`, `topic_number`, `topic_title`, `subject_number`, `subject_title`, `article_anchor`, `article_title`, `chapter_title`, `source_note_text`, `source_links`, `related_note_text`, `content_text`, `content_char_len`, `content_word_count`, `source_url`, `scraped_at` | 7 parquet files (large) | **Corpus chính** — đã chunk sẵn theo Điều, có `topic_title` để lọc SME, có `source_links` |
| `adamwhite625/vietnam-legal-qa` | QA pairs | `id`, `question`, `law_content`, `law_id`, `law_name` | Large | **Eval & few-shot** — QA sẵn có, dùng để đánh giá & prompt |

- `UTS_VLC`: Full markdown docs, ID = mã luật (VD: "91/2015/QH13"), content = toàn văn → cần parse & chunk
- `tmquan/phapdien-moj-gov-vn`: Đã chunk sẵn theo Điều, có `topic_title` để lọc SME (VD: "Doanh nghiệp", "Lao động"), có `chapter_title`, `content_text`, `source_links` → **không cần chunk thêm, chỉ cần lọc theo topic**
- `adamwhite625/vietnam-legal-qa`: QA pairs có sẵn law_content + law_name → dùng cho few-shot prompting & đánh giá
- Bổ sung: `uitnlp/ALQAC`, `Truong-Phan/ViLegal`, `nqminh0106/VietLaw`

**1.2. Lọc corpus theo chủ đề SME**

> **KHÔNG lọc chỉ lấy "luật doanh nghiệp"** — F2 ưu tiên Recall 2x, thiếu điều luật bị phạt nặng hơn trả thừa.

Lọc UTS_VLC + phapdien theo danh sách ~50-100 văn bản liên quan SME:

**Bắt buộc (core SME)**:
- Luật Doanh nghiệp 59/2020/QH14
- Luật Hỗ trợ DN nhỏ và vừa 04/2017/QH14
- Luật Lao động 45/2019/QH14
- Luật Quản lý thuế 38/2019/QH14
- Bộ luật Dân sự 91/2015/QH13
- Luật Đầu tư 61/2020/QH14

**Nên có (Nghị định/Thông tư hướng dẫn)**:
- NĐ-CP: 01/2021, 80/2021, 145/2020, 123/2020, 94/2023
- Thông tư: đăng ký doanh nghiệp, thuế, lao động

**Có thể thêm (mở rộng recall)**:
- Luật Thuế TNDN, Thuế GTGT, Bảo hiểm xã hội, Thương mại, Cạnh tranh

**1.3. Scrape vbpl.vn (bổ sung nếu thiếu)**
- Chỉ scrape văn bản thiếu trong HuggingFace datasets
- Dùng requests + BeautifulSoup, rate-limit 1-2 giây

**1.3. Parse & chunk văn bản**
- Parse HTML → text có cấu trúc
- Chunk theo cấp Điều — phù hợp với đánh giá cuộc thi
- Mỗi chunk có metadata: `law_id`, `law_name`, `article_num`, `chapter`, `text`
- Format `law_name`: `Loại văn bản + Mã văn bản + Trích yếu`
- Điều dài → split tại Khoản, giữ reference Điều cha

**1.4. Gộp & loại trùng**
- Merge HuggingFace + scraped data
- Deduplicate theo cặp (law_id, article_num)
- Output: `data/processed/articles.jsonl`

---

### Giai đoạn 2: Indexing & Retrieval (Ngày 4-6)

**2.1. Build FAISS dense index**
- Encode tất cả articles bằng `BAAI/bge-m3` (dense + sparse)
- Lưu dense vectors vào FAISS (IndexFlatIP hoặc IndexIVFFlat)
- Xử lý batch trên GPU (24GB VRAM đủ thoải mái)

**2.2. Build BM25 index**
- Tokenize tiếng Việt bằng `underthesea` hoặc `pyvi`
- Build BM25 index bằng `rank_bm25`
- Bắt chính xác mã luật (VD: "04/2017/QH14") mà semantic search có thể bỏ sót

**2.3. Hybrid retrieval (RRF Fusion)**
- Dense retrieval: top-50 từ FAISS (cosine similarity)
- Sparse retrieval: top-50 từ BM25
- Merge bằng Reciprocal Rank Fusion: `score(d) = Σ 1/(k + rank_i(d))`, k=60

**2.4. Cross-encoder reranking**
- Load `BAAI/bge-reranker-v2-m3`
- Rerank top-50 hybrid → top-10
- Cải thiện Precision đáng kể mà vẫn giữ Recall cao

---

### Giai đoạn 3: QA Generation (Ngày 7-10)

**3.1. Load LLMs**
- Dùng vLLM hoặc transformers cho cả Vistral-7B và Qwen2.5-7B
- 4-bit quantization (bitsandbytes) nếu cần tiết kiệm VRAM

**3.2. Prompt templates**
- System: vai trò trợ lý pháp lý, phải trích dẫn điều luật, cảnh báo giới hạn AI
- User: câu hỏi + context (các điều luật đã truy hồi)
- Yêu cầu format: "Theo Điều X Luật Y..." trong câu trả lời
- Bao gồm few-shot examples

**3.3. Sinh câu trả lời**
- Với mỗi câu hỏi test:
  1. Truy hồi top-10 articles qua hybrid + rerank
  2. Format context kèm metadata
  3. Sinh answer bằng LLM
  4. Post-process: trích xuất điều luật từ answer text
  5. Build `relevant_docs` và `relevant_articles`

**3.4. Trích xuất & validate citations**
- Parse answer text tìm pattern "Điều X"
- Match với retrieved articles theo law_id + article_num
- Validate citations tồn tại trong corpus
- Điền `relevant_docs` và `relevant_articles` theo format cuộc thi

---

### Giai đoạn 4: Tối ưu & Nộp bài (Ngày 11-14)

**4.1. So sánh LLMs**
- Chạy pipeline với cả Vistral-7B và Qwen2.5-7B trên sample
- So sánh chất lượng answer, độ tự nhiên tiếng Việt, citation accuracy
- Chọn model tốt hơn cho submission

**4.2. Tune retrieval parameters**
- Điều chỉnh: top-k dense, top-k BM25, RRF k, rerank top-n
- Mục tiêu: maximize F2 → ưu tiên Recall cao
- Có thể tăng candidates (top-100) rồi rely on reranking

**4.3. Cải thiện chunking**
- Nếu retrieval miss articles:
  - Chunk tại Khoản level (mịn hơn)
  - Thêm context Chương vào chunk text
  - Thêm tiêu đề/summary điều luật vào embedding

**4.4. Tạo bài nộp**
- Chạy pipeline trên toàn bộ test set
- Output `results.json` đúng format
- Validate bằng `validate.py`
- Zip & submit

---

### Giai đoạn 5: Iteration (Ngày 15-21)

- Phân tích feedback từ leaderboard
- Fine-tune embedding model trên legal QA pairs (nếu còn thời gian)
- Cải thiện prompt engineering
- Thêm multi-hop retrieval cho câu hỏi phức tạp
- Nộp bài cuối cùng trước 30/06

---

## Dependencies chính

```
torch>=2.0
transformers>=4.36
accelerate
bitsandbytes
FlagEmbedding          # BGE-M3 + reranker
faiss-gpu              # hoặc faiss-cpu
rank-bm25
datasets               # HuggingFace datasets
beautifulsoup4
playwright             # scraping
underthesea            # Vietnamese NLP
pyvi                   # Vietnamese word segmentation
pyyaml
tqdm
```

---

## Định dạng output (results.json)

```json
[{
  "id": 1,
  "question": "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào...",
  "answer": "Theo Điều 4 Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa, doanh nghiệp được hỗ trợ khi...",
  "relevant_docs": [
    "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
  ],
  "relevant_articles": [
    "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4",
    "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 5"
  ]
}]
```

**Lưu ý quan trọng**:
- `law_name` phải theo công thức: **Loại văn bản + Mã văn bản + Trích yếu**
- Hệ thống chấm tự động trích xuất "Điều X" từ trường `answer`
- File zip chỉ chứa `results.json` ở gốc, không nằm trong thư mục con