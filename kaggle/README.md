# Kaggle Pipeline - Vietnamese Legal RAG

Chạy từng giai đoạn riêng lẻ. Mỗi giai đoạn có input/output rõ ràng, chỉ cần chạy lại giai đoạn muốn thay đổi.

## Tổng quan

```
Stage 1: Data     Stage 2: BM25      Stage 3: Dense       Stage 4: Retrieval
(2 min)           (1.5h)             (3h on T4)           (30-60 min)
    │                 │                   │                     │
    ▼                 ▼                   ▼                     ▼
articles.jsonl   bm25_index.pkl    index.faiss           results.json
                                   metadata.jsonl        submission.zip
```

## Chi tiết từng giai đoạn

### Stage 1: Data Collection & Processing
| | |
|---|---|
| **Script** | `1_data.py` |
| **Input** | Không (download từ HuggingFace) |
| **Output** | `articles.jsonl` |
| **Thời gian** | ~2 phút |
| **GPU** | Không cần |
| **Lưu Kaggle Dataset** | `r2ai-articles` |

```bash
!pip install -q datasets pyvi tqdm
!python 1_data.py
```

**Fix quan trọng**: law_id normalization cho UTS_VLC — resolve slug IDs thành chuẩn format, skip docs không resolve được.

---

### Stage 2: BM25 Indexing
| | |
|---|---|
| **Script** | `2_bm25.py` |
| **Input** | `articles.jsonl` (từ Stage 1) |
| **Output** | `bm25_index.pkl` |
| **Thời gian** | ~1.5 giờ |
| **GPU** | Không cần |
| **Lưu Kaggle Dataset** | `r2ai-bm25` |

```bash
!pip install -q rank-bm25 pyvi tqdm
!python 2_bm25.py
```

---

### Stage 3: Dense Embeddings + FAISS
| | |
|---|---|
| **Script** | `3_dense.py` |
| **Input** | `articles.jsonl` (từ Stage 1) |
| **Output** | `index.faiss`, `metadata.jsonl`, `dense.npy` |
| **Thời gian** | ~3 giờ trên T4 |
| **GPU** | Cần (T4/P100) |
| **Lưu Kaggle Dataset** | `r2ai-faiss` |

```bash
!pip install -q sentence-transformers faiss-cpu tqdm
!python 3_dense.py
```

Có checkpoint mỗi 5000 items, nếu bị ngắt thì chạy lại sẽ tiếp tục.

---

### Stage 4: Retrieval + Submission
| | |
|---|---|
| **Script** | `4_retrieval.py` |
| **Input** | `articles.jsonl`, `bm25_index.pkl`, `index.faiss`, `metadata.jsonl`, `R2AIStage1DATA.json` |
| **Output** | `results.json`, `submission.zip` |
| **Thời gian** | 30-60 phút |
| **GPU** | Cần cho reranker (có thể skip) |

```bash
!pip install -q sentence-transformers rank-bm25 pyvi faiss-cpu tqdm
!python 4_retrieval.py
```

**Options**:
```bash
!python 4_retrieval.py --no-rerank      # Skip reranker (nhanh hơn)
!python 4_retrieval.py --bm25-only       # Chỉ BM25, không cần FAISS
!python 4_retrieval.py --limit 50        # Chỉ chạy 50 câu (debug)
```

---

## Cách chạy trên Kaggle

### Lần đầu (chạy hết 4 stages)

1. Tạo Kaggle Notebook mới, bật GPU
2. Upload các script `.py` + `R2AIStage1DATA.json` vào Kaggle dataset
3. Chạy từng stage:
```python
# Cell 1: Setup
!pip install -q datasets sentence-transformers rank-bm25 pyvi faiss-cpu tqdm

# Cell 2: Stage 1 - Data
!python 1_data.py

# Cell 3: Stage 2 - BM25
!python 2_bm25.py

# Cell 4: Stage 3 - Dense
!python 3_dense.py

# Cell 5: Stage 4 - Retrieval
!python 4_retrieval.py
```

### Từ lần 2 (đã có output từ stages trước)

1. Lưu output của mỗi stage thành Kaggle Dataset riêng:
   - `r2ai-articles` → chứa `articles.jsonl`
   - `r2ai-bm25` → chứa `bm25_index.pkl`
   - `r2ai-faiss` → chứa `index.faiss`, `metadata.jsonl`

2. Khi tạo notebook mới, Add Data → chọn các datasets trên
   → Files sẽ có sẵn tại `/kaggle/input/r2ai-*/`

3. Copy files vào working directory:
```python
import shutil, glob
for f in glob.glob('/kaggle/input/r2ai-articles/*'):
    shutil.copy(f, '/kaggle/working/')
for f in glob.glob('/kaggle/input/r2ai-bm25/*'):
    shutil.copy(f, '/kaggle/working/')
for f in glob.glob('/kaggle/input/r2ai-faiss/*'):
    shutil.copy(f, '/kaggle/working/')
```

4. Chỉ chạy stage cần thay đổi (thường là Stage 4)

### Chỉ test retrieval (nhanh nhất)

Nếu chỉ muốn test tham số retrieval mà không cần dense embeddings:
```bash
!python 4_retrieval.py --bm25-only --limit 50
```

Chỉ cần `articles.jsonl` + `bm25_index.pkl`, chạy 50 câu trong ~2 phút.
