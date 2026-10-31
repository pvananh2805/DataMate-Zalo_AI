# Tích hợp model local vào DataMate

Tài liệu này hướng dẫn nối một checkpoint đã được bạn lựa chọn và chuẩn bị vào
pipeline inference. DataMate **chưa chọn hoặc phân phối model cụ thể**; các bước
dưới đây không khẳng định checkpoint nào đã được benchmark hoặc phù hợp điều lệ.

## Model được dùng ở bước nào?

Model chỉ chuyển câu hỏi nghiệp vụ thành `QueryPlan` JSON. DataMate kiểm tra
plan, đọc dữ liệu doanh nghiệp và để executor xác định tính toán câu trả lời.
Model không trực tiếp truy cập cơ sở dữ liệu hoặc trả kết quả cuối thay executor.

Đường chạy hiện tại:

```text
câu hỏi + schema + context
        → Transformers local checkpoint
        → QueryPlan JSON
        → validator → executor → answer/evidence/trace
```

Entrypoint là `scripts/run_inference.py`; lớp nạp model là
`src/datamate/local_inference.py`.

## Điều kiện tương thích

Checkpoint cần:

- Là model causal language model có thể nạp bằng
  `transformers.AutoModelForCausalLM` và tokenizer bằng
  `transformers.AutoTokenizer`.
- Có đủ tokenizer, config và weights trong thư mục local.
- Không phụ thuộc `trust_remote_code=True`; backend chủ động đặt
  `trust_remote_code=False`.
- Có thể sinh tiếp văn bản từ prompt. Chat template được dùng nếu tokenizer có
  khai báo; nếu không, backend gửi prompt dạng text thường.
- Vừa VRAM của GPU ở precision bạn chọn và có không quá `parameter_cap` tham số
  (mặc định 15 tỷ).

Model instruction/chat thường phù hợp hơn model base cho sinh JSON, nhưng đây
chỉ là tiêu chí lựa chọn cần benchmark, không phải xác nhận một model cụ thể.
Một checkpoint tương thích API vẫn có thể tạo QueryPlan không đúng ngữ nghĩa.

## 1. Chuẩn bị checkpoint

Tải/chuẩn bị checkpoint theo giấy phép, nguồn và quy định áp dụng. Khi hoàn tất,
đặt thư mục model ở nơi local; ví dụ:

```text
D:\DataMate-ZaloAI\models\my-checkpoint\
    config.json
    tokenizer.json / tokenizer.model / tokenizer_config.json
    model.safetensors hoặc shard weights và index tương ứng
```

Tên file thực tế tùy checkpoint. Nếu dùng LoRA, backend này không nạp adapter
riêng: trước tiên cần merge/export thành model đầy đủ tương thích
`AutoModelForCausalLM`. Không commit checkpoint, token truy cập hoặc thông tin
tài khoản vào repository.

## 2. Cài runtime trên máy GPU

Tạo môi trường ảo rồi cài **bản PyTorch đúng với GPU/driver/CUDA của máy** theo
hướng dẫn chính thức của PyTorch. Sau đó cài dependency inference:

```powershell
python -m pip install -r requirements-inference.txt
```

Trên Linux, dùng cùng lệnh trong virtual environment Linux.
`requirements-inference.txt` khai báo dải phiên bản torch/transformers nhưng
không chọn build CUDA cho máy của bạn. Không dùng lệnh cài PyTorch chung một
cách mù quáng nếu máy cần build CUDA cụ thể.

Kiểm tra runtime:

```powershell
python -c "import torch, transformers; print(torch.__version__); print(transformers.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Inference hiện từ chối chạy nếu CUDA không khả dụng; không có CPU fallback.
Precision mặc định là `bf16`, chỉ chạy nếu GPU hỗ trợ. Nếu GPU hỗ trợ FP16 nhưng
không hỗ trợ BF16, có thể cấu hình `fp16`. Không thể suy ra VRAM cần thiết chỉ
từ tên model; hãy đo trên checkpoint, prompt length và GPU thực tế.

## 3. Khai báo đường dẫn model

Tạo config local từ mẫu:

```powershell
Copy-Item config\inference.example.json config\inference.local.json
```

Trong `config\inference.local.json`, sửa tối thiểu:

```json
{
  "input": "..\\input\\questions.jsonl",
  "business_data_root": "..\\input\\business",
  "model_path": "D:\\DataMate-ZaloAI\\models\\my-checkpoint",
  "log": "..\\output\\run.jsonl",
  "submission_output": "..\\output\\submission.jsonl",
  "precision": "bf16",
  "max_new_tokens": 1024,
  "parameter_cap": 15000000000
}
```

Đây là các trường cần thay trong config mẫu; giữ lại các trường context khác
nếu cần. `model_path` có thể là đường dẫn tuyệt đối hoặc tương đối so với thư
mục chứa config. Config cá nhân `config/inference.local.json` được Git ignore.
Các thư mục input/output phải tồn tại phù hợp; output không được nằm trong
input/data root.

## 4. Chạy inference

Từ thư mục gốc repository:

```powershell
python scripts\run_inference.py --config config\inference.local.json
```

Hoặc kiểm tra CLI:

```powershell
python scripts\run_inference.py --help
```

Nếu chỉ đổi precision hoặc giới hạn sinh, có thể override bằng flags, ví dụ:

```powershell
python scripts\run_inference.py --config config\inference.local.json --precision fp16 --max-new-tokens 512
```

CLI nạp model một lần, xử lý câu hỏi JSONL tuần tự và ghi run log cùng submission
output. Submission hiện theo schema tham khảo training rulebook; chưa xác nhận
với validator chính thức. Nếu lỗi một câu hỏi, process trả exit code khác 0 và
không xuất submission một phần.

## 5. Kiểm tra model bằng manual questions

Tự viết câu hỏi trong `manual_test/questions.jsonl` rồi chạy:

```powershell
python scripts\run_manual_test.py --business-data-root <business-root> --model-path "D:\DataMate-ZaloAI\models\my-checkpoint"
```

`manual_test/README.md` mô tả schema câu hỏi/expected answer và file kết quả.
Chỉ tự thêm đáp án chuẩn nếu bạn có đáp án đáng tin cậy; không để đáp án chuẩn
lọt vào prompt planner. Bắt đầu với một vài câu hỏi tự tạo và kiểm tra
QueryPlan, evidence, trace cùng answer; test tự động dùng planner stub không
thay thế bước kiểm tra checkpoint thật.

## Offline, an toàn và chẩn đoán

Sau khi đã cài packages và chuẩn bị checkpoint, backend đặt
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, dùng `local_files_only=True` và
không gọi API inference. Cài dependency hoặc lấy checkpoint trước đó có thể
cần mạng; đó là bước chuẩn bị riêng. Đường chạy inference không tự tải model.

| Thông báo/tình trạng | Kiểm tra |
|---|---|
| `model_path must be a local checkpoint directory` | Đường dẫn config phải trỏ đến thư mục checkpoint đã tồn tại. |
| Thiếu PyTorch hoặc Transformers | Cài PyTorch CUDA phù hợp trước, rồi cài inference requirements. |
| `Local Transformers inference is configured for CUDA` | Kiểm tra driver, CUDA build của PyTorch và `torch.cuda.is_available()`. |
| GPU không hỗ trợ `bf16` | Chỉ đổi sang `fp16` nếu GPU hỗ trợ và checkpoint chạy ổn với precision đó. |
| Không tìm thấy config/tokenizer/weights | Chuẩn bị đầy đủ artifact của checkpoint; backend không tải phần còn thiếu. |
| Vượt parameter cap hoặc hết VRAM | Kiểm tra đúng checkpoint, precision, VRAM; không tăng cap nếu chưa xác nhận tài nguyên. |
| QueryPlan sai hoặc JSON không hợp lệ | Xem run log/manual results. Khả năng tạo plan đúng phải benchmark riêng cho checkpoint/model prompt. |

Tài liệu này mô tả đường tích hợp đã có trong source, không chứng minh model
thực tế đã chạy. Trạng thái GPU/checkpoint/submission được cập nhật tại
[planning/17_DEPLOYMENT_REPORT.md](planning/17_DEPLOYMENT_REPORT.md).
