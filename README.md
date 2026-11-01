# DataMate - Zalo AI Challenge

DataMate là giải pháp toàn diện cho bài toán **Text-to-QueryPlan & Deterministic Business Intelligence**, chuyển đổi các câu hỏi nghiệp vụ tiếng Việt phức tạp thành `QueryPlan` khai báo chuẩn tắc, thực thi có kiểm soát trên dữ liệu doanh nghiệp cục bộ (Excel, CSV, SQLite), và trả về câu trả lời chính xác cùng bằng chứng (evidence) và dấu vết thực thi (trace).

Hệ thống tuân thủ nghiêm ngặt nguyên tắc **solve-first & fail-closed**:
- Mọi câu hỏi phải được diễn giải thành `QueryPlan` hợp lệ, được chuẩn hóa và kiểm tra schema trước khi thực thi.
- Không cho phép thực thi mã tùy ý (arbitrary code) hoặc SQL injection trực tiếp từ câu hỏi.
- Khi dữ liệu không đủ, câu hỏi mập mờ hoặc vi phạm ràng buộc nghiệp vụ, hệ thống từ chối an toàn (`abstain`) thay vì phán đoán bừa bãi.

---

## 1. Kiến trúc Pipeline

```text
               Câu hỏi tiếng Việt (questions.jsonl)
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │       1. Question Understanding           │
        │   - Trích xuất ý định nghiệp vụ           │
        │   - Qwen2.5 Local Foundation Model        │
        │   - JSON Repair & Safe Auto-Healing       │
        └─────────────────────┬─────────────────────┘
                              │ QueryPlan JSON
                              ▼
        ┌───────────────────────────────────────────┐
        │       2. Plan & Schema Validation         │
        │   - Đối chiếu danh mục bảng, cột          │
        │   - Kiểm tra kiểu dữ liệu, phép kết hợp   │
        │   - Ràng buộc an toàn & quy tắc nghiệp vụ │
        └─────────────────────┬─────────────────────┘
                              │ Validated Plan
                              ▼
        ┌───────────────────────────────────────────┐
        │       3. Deterministic Query Executor     │
        │   - Multi-sheet Excel / CSV / SQLite      │
        │   - Join, Group By, Filter, Aggregation   │
        │   - Xử lý thời gian (Rulebook Calendar)   │
        │   - Tính toán tỷ lệ, làm tròn chuẩn hóa   │
        └─────────────────────┬─────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    submission.jsonl                    run.jsonl (Audit Trace)
    (Đáp án nộp bài)                    (Bằng chứng & Nhật ký chi tiết)
```

- **Question Understanding** (`src/datamate/question_understanding.py`): Tiếp nhận câu hỏi, ngữ cảnh lịch, rulebook và schema; sử dụng LLM cục bộ để tạo `QueryPlan` có cấu trúc JSON; tự động sửa chữa (auto-healing) các sai lệch về alias bảng/năm, thứ tự cột và kiểu đầu ra.
- **Query Executor** (`src/datamate/query_executor.py`): Thực thi truy vấn hoàn toàn tất định (deterministic), hỗ trợ join đa bảng, lọc thời gian theo ca (sáng/chiều), nhóm, tính tổng/đếm/trung bình, và xử lý các quy tắc loại trừ đơn hủy.
- **Data Loader** (`src/datamate/data_loader.py`): Đọc và chuẩn hóa dữ liệu từ nhiều nguồn `.xlsx`, `.csv`, `.sqlite`.

---

## 2. Yêu cầu hệ thống

- **Hệ điều hành**: Windows 10/11 hoặc Linux (Ubuntu 20.04+).
- **Python**: Phiên bản **3.10 - 3.12** (khuyến nghị Python **3.12**).
- **Phần cứng**:
  - **GPU**: NVIDIA GPU có hỗ trợ CUDA (khuyến nghị 6GB VRAM trở lên, hỗ trợ `bf16` hoặc `fp16`). Đã kiểm thử thành công trên GTX 1060 (6GB) và các dòng RTX.
  - **CPU**: Có thể chạy hoàn toàn trên CPU với `--device cpu --precision fp32`.

---

## 3. Cài đặt môi trường (Setup)

### Bước 1: Clone kho mã nguồn
```bash
git clone https://github.com/pvananh2805/DataMate-Zalo_AI.git
cd DataMate-Zalo_AI
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Virtual Environment)

- **Trên Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

- **Trên Linux / macOS (Bash)**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Bước 3: Cài đặt PyTorch với CUDA
Tùy vào phiên bản CUDA của máy, cài đặt PyTorch phù hợp từ trang chủ PyTorch:
```bash
# Ví dụ với CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Hoặc phiên bản chỉ dùng CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Bước 4: Cài đặt các gói phụ thuộc
```bash
# Cài đặt thư viện cho inference
pip install -r requirements-inference.txt

# Cài đặt thư viện dev / testing (tùy chọn)
pip install -r requirements-dev.txt
```

---

## 4. Chuẩn bị Model & Dữ liệu

### 4.1. Chuẩn bị Checkpoint Model
DataMate chạy **hoàn toàn offline**, không gửi dữ liệu ra bên ngoài và không tự động tải model từ internet lúc inference. Bạn cần tải checkpoint mô hình (ví dụ `Qwen/Qwen2.5-3B-Instruct` hoặc `Qwen/Qwen2.5-0.5B-Instruct`) và lưu vào một thư mục local:
```text
C:\Users\...\models\foundation\qwen2.5_3b\
├── config.json
├── model.safetensors (hoặc các file shards)
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
```

### 4.2. Cấu trúc thư mục dữ liệu
Dữ liệu doanh nghiệp được đặt trong thư mục `business_data_root`:
```text
data/public_test/
├── calendar.csv               # Bảng tra cứu lịch, tuần, tháng
├── rulebook.md                # Quy tắc nghiệp vụ cuộc thi
├── questions.jsonl            # Tập câu hỏi đầu vào
├── dapan.jsonl                # Bộ đáp án chuẩn (ground truth)
└── data/                      # Dữ liệu bảng của từng doanh nghiệp
    ├── ChiTietHoaDon_1.xlsx
    ├── BanHang_TongHop_2026.xlsx
    ├── DanhSachTho.xlsx
    └── SoSuaXe.xlsx
```

---

## 5. Cấu hình chi tiết (`config/*.json`)

Các thông số vận hành được quản lý tập trung trong file JSON. Ví dụ file cấu hình cho 10 câu test public: `config/inference.10_test.json`:

```json
{
  "input": "data/public_test/questions_10.jsonl",
  "business_data_root": "data/public_test/data",
  "model_path": "C:\\path\\to\\qwen2.5_3b",
  "device": "cuda",
  "precision": "bf16",
  "log": "output/public_test/run_10.jsonl",
  "submission_output": "output/public_test/submission_10.jsonl",
  "rulebook": "data/public_test/rulebook.md",
  "calendar": "data/public_test/calendar.csv",
  "coercions": null,
  "as_of_date": null,
  "max_prompt_tokens": 3072,
  "max_new_tokens": 512,
  "parameter_cap": 15000000000,
  "generation_timeout_seconds": null
}
```

### Giải thích các tham số cấu hình:
| Tham số | Kiểu | Ý nghĩa |
| :--- | :--- | :--- |
| `input` | `str` | Đường dẫn file JSONL chứa các câu hỏi cần trả lời. |
| `business_data_root` | `str` | Thư mục gốc chứa các bảng dữ liệu (.xlsx, .csv, .sqlite). |
| `model_path` | `str` | Đường dẫn local đến thư mục chứa trọng số mô hình LLM. |
| `device` | `str` | Thiết bị chạy: `"cuda"` (GPU NVIDIA) hoặc `"cpu"`. |
| `precision` | `str` | Độ chính xác trọng số: `"bf16"` (khuyến nghị GPU mới), `"fp16"`, hoặc `"fp32"` (CPU). |
| `submission_output` | `str` | Đường dẫn xuất file nộp bài định dạng JSONL (`id`, `type`, `answer`). |
| `log` | `str` | Đường dẫn xuất file log chi tiết chứa QueryPlan, evidence và trace. |
| `rulebook` | `str` | Đường dẫn file markdown quy tắc nghiệp vụ (`rulebook.md`). |
| `calendar` | `str` | Đường dẫn file lịch (`calendar.csv`) tra cứu mốc ngày, tuần. |
| `max_prompt_tokens`| `int` | Giới hạn số token đầu vào của prompt (mặc định: 3072). |
| `max_new_tokens` | `int` | Giới hạn số token tối đa mô hình sinh ra (mặc định: 512). |
| `parameter_cap` | `int` | Giới hạn số tham số tối đa cho phép của mô hình (quy định: 15 tỷ). |

---

## 6. Hướng dẫn chạy và đánh giá (Execution Guide)

### 6.1. Chạy Inference (Sinh đáp án)

- **Trên Windows PowerShell**:
  ```powershell
  $env:PYTHONPATH = "src"
  python scripts/run_inference.py --config config/inference.10_test.json
  ```

- **Trên Linux / Bash**:
  ```bash
  PYTHONPATH=src python scripts/run_inference.py --config config/inference.10_test.json
  ```

Khi hoàn tất, file đáp án sẽ xuất hiện tại `output/public_test/submission_10.jsonl` và file log chi tiết tại `output/public_test/run_10.jsonl`.

### 6.2. Chấm điểm kết quả tự động (Evaluation)
Sử dụng script đánh giá đối chiếu với bộ đáp án ground truth `data/public_test/dapan.jsonl`:
```bash
python scripts/evaluate_submission.py output/public_test/submission_10.jsonl
```

Kết quả hiển thị chi tiết từng câu hỏi (CORRECT, WRONG, ABSTAIN) và tổng điểm.
> Hiện tại pipeline đạt điểm tuyệt đối **10/10 (100% Correct, 0 Wrong, 0 Abstain sai)** trên bộ 10 câu Public Test.

### 6.3. Chạy kiểm thử tự động (Unit & Integration Tests)
Chạy toàn bộ 250 bài test hồi quy để đảm bảo các thay đổi không làm gãy logic hệ thống:
```bash
# Windows PowerShell
$env:PYTHONPATH = "src"; python -m pytest tests/

# Linux / Bash
PYTHONPATH=src pytest tests/
```

### 6.4. Replay nhanh lượt chạy từ Log
Để tái hiện hoặc debug kết quả thực thi của một lượt chạy mà **không cần gọi lại model LLM**:
```bash
python scripts/replay_run.py --log output/public_test/run_10.jsonl
```

---

## 7. Giải pháp tối ưu tốc độ thực thi (Làm cách nào để nhanh hơn?)

Khi chạy tập dữ liệu lớn với hàng trăm câu hỏi và các file Excel lên tới 100.000 dòng, thời gian thực thi có thể bị kéo dài. Dưới đây là phân tích các **nút thắt cổ chai (bottlenecks)** chính và các giải pháp tối ưu hóa hiệu năng cao nhất:

### 🚀 Nút thắt 1: Tốc độ đọc file Excel (.xlsx) cỡ lớn
- **Vấn đề**: Hiện tại `openpyxl` duyệt qua từng ô của file Excel 50.000 – 100.000 dòng mất từ **30 đến 60 giây mỗi câu hỏi**.
- **Giải pháp tối ưu**:
  1. **Chuyển đổi / Pre-cache sang Parquet hoặc DuckDB**:
     - Trước khi chạy batch, thực hiện một bước quét nhanh (pre-flight): đọc tất cả các file `.xlsx` một lần duy nhất và lưu cache thành `.parquet` hoặc database in-memory DuckDB / SQLite có đánh index.
     - Tốc độ đọc Parquet / DuckDB chỉ mất **0.01 – 0.05 giây** (nhanh gấp **1000 lần** so với đọc trực tiếp Excel).
  2. **Sử dụng Rust Excel Engine (`python-calamine`)**:
     - Thư viện `python-calamine` được viết bằng Rust, tốc độ đọc file Excel nhanh gấp **15 đến 20 lần** `openpyxl`.
  3. **Bật chế độ `read_only=True, data_only=True`**:
     - Trong `openpyxl`, thiết lập `read_only=True` giúp stream dữ liệu từ ổ đĩa thay vì nạp toàn bộ DOM cây XML vào RAM.

### ⚡ Nút thắt 2: Tốc độ sinh của mô hình LLM (Inference Latency)
- **Vấn đề**: Việc sinh tuần tự từng token với Hugging Face Transformers truyền thống có thể mất 3-8 giây mỗi câu hỏi.
- **Giải pháp tối ưu**:
  1. **Chuyển sang vLLM hoặc SGLang (Khuyến nghị số 1 trên Linux/WSL)**:
     - vLLM sử dụng cơ chế **PagedAttention** và **Continuous Batching**, giúp tăng tốc độ sinh lên **5x – 10x** so với Transformers chuẩn, tối ưu hóa triệt để VRAM của GPU.
  2. **Bật FlashAttention-2 & Compilation**:
     - Khởi tạo model với `attn_implementation="flash_attention_2"` (trên GPU Ampere / Ada Lovelace / Hopper / Blackwell).
     - Áp dụng `torch.compile(model)` trên PyTorch 2.x để gom các kernel tính toán.
  3. **Giảm `max_new_tokens` và cấu hình Early Stopping**:
     - Cấu trúc `QueryPlan` chuẩn chỉ cần khoảng **150 – 250 tokens**. Giảm `max_new_tokens` từ 512 xuống 256.
     - Thêm `StoppingCriteria` dừng ngay lập tức khi mô hình sinh ra dấu ngoặc nhọn kết thúc JSON `\n}` để tránh thời gian sinh thừa.
  4. **Lượng tử hóa (Quantization)**:
     - Chạy mô hình ở định dạng **4-bit AWQ** hoặc **BitsAndBytes NF4**. Việc này giảm dung lượng đọc/ghi trên VRAM băng thông GPU, giúp tốc độ sinh token tăng từ 1.5x đến 2x mà độ chính xác gần như không suy giảm.

### ✂️ Nút thắt 3: Độ dài Context / Prompt (Prompt Pruning & Retrieval)
- **Vấn đề**: Hiện tại toàn bộ schema của tất cả các bảng trong thư mục doanh nghiệp cùng với rulebook và calendar được nhồi vào prompt (~3.000 tokens). Thời gian xử lý prompt ban đầu (Time-to-First-Token - TTFT) chiếm phần lớn độ trễ.
- **Giải pháp tối ưu**:
  1. **Schema Retrieval (Lọc schema liên quan)**:
     - Dùng từ khóa từ câu hỏi hoặc thuật toán BM25 nhẹ để chỉ lọc ra **1 đến 2 bảng liên quan nhất** đưa vào prompt thay vì đưa toàn bộ 10-20 bảng.
     - Rút ngắn context từ 3.000 tokens xuống còn **500 – 800 tokens**, giúp TTFT giảm đi **4 lần**.
  2. **Nén Context Lịch (Calendar Slicing)**:
     - Chỉ truyền vào các dòng lịch xung quanh mốc `as_of_date` hoặc khoảng năm xuất hiện trong câu hỏi thay vì toàn bộ file `calendar.csv`.

### 🔄 Nút thắt 4: Kiến trúc Xử lý Song song (Async / Multi-threading Pipeline)
- **Vấn đề**: Hiện tại pipeline chạy tuần tự: Model sinh câu 1 -> Executor thực thi câu 1 -> Model sinh câu 2 -> Executor thực thi câu 2. GPU phải ngồi chờ khi CPU đọc đĩa, và ngược lại.
- **Giải pháp tối ưu**:
  - Áp dụng mô hình **Producer - Consumer (Đa luồng bất đồng bộ)**:
    - **Luồng 1 (GPU Producer)**: Liên tục sinh QueryPlan cho các câu hỏi tiếp theo và đẩy vào hàng đợi (Queue).
    - **Luồng 2 (CPU Consumer)**: Lấy QueryPlan từ hàng đợi, thực hiện nạp dữ liệu và tính toán kết quả trên các lõi CPU song song.
  - Nhờ đó, GPU và CPU luôn hoạt động đồng thời 100% công suất, giảm thời gian tổng của cả pipeline xuống gần một nửa.

---

## 8. Cấu trúc thư mục dự án

```text
DataMate-Zalo_AI/
├── config/                         # Các file cấu hình mẫu và cấu hình chạy
│   ├── inference.10_test.json      # Cấu hình kiểm thử 10 câu Public Test
│   └── inference.example.json      # File mẫu hướng dẫn cấu hình
├── data/                           # Thư mục dữ liệu kiểm thử và benchmark
│   └── public_test/
│       ├── dapan.jsonl             # Đáp án chuẩn 10 câu Public Test
│       ├── questions_10.jsonl      # Danh sách câu hỏi kiểm thử
│       ├── calendar.csv            # Lịch và ánh xạ tuần/tháng
│       └── rulebook.md             # Quy tắc nghiệp vụ cuộc thi
├── scripts/                        # Scripts thực thi chính
│   ├── run_inference.py            # Entry point chạy toàn bộ pipeline
│   ├── evaluate_submission.py      # Script chấm điểm tự động đối chiếu đáp án
│   ├── replay_run.py               # Replay lượt chạy từ file log
│   └── run_manual_test.py          # Chạy kiểm thử câu hỏi tùy biến
├── src/datamate/                   # Mã nguồn cốt lõi
│   ├── business_context.py         # Nạp ngữ cảnh rulebook & calendar
│   ├── data_loader.py              # Đọc file dữ liệu bảng đa định dạng
│   ├── local_inference.py          # Quản lý nạp model và sinh token offline
│   ├── pipeline.py                 # Điều phối luồng dữ liệu end-to-end
│   ├── plan_serialization.py       # Tuần tự hóa và kiểm tra cấu trúc QueryPlan
│   ├── query_executor.py           # Bộ thực thi tính toán tất định
│   └── question_understanding.py   # Phân tích câu hỏi và Auto-healing QueryPlan
├── tests/                          # 250 bài test hồi quy tự động
│   ├── integration/                # Test tích hợp luồng
│   └── unit/                       # Test chức năng từng module
├── requirements-inference.txt      # Thư viện phục vụ inference
├── requirements-dev.txt            # Thư viện phục vụ test và development
└── README.md                       # Hướng dẫn chi tiết
```

---

## 9. Giấy phép & Đóng góp
Dự án được xây dựng cho thử thách Zalo AI Challenge. Mọi đóng góp, báo lỗi hoặc đề xuất cải tiến vui lòng mở Issue hoặc Pull Request trên kho lưu trữ [GitHub](https://github.com/pvananh2805/DataMate-Zalo_AI).
