# DataMate project for ZALO AI

DataMate là một pipeline thử nghiệm để chuyển câu hỏi nghiệp vụ tiếng Việt thành
`QueryPlan`, đọc dữ liệu doanh nghiệp cục bộ, thực thi truy vấn có kiểm soát và
ghi câu trả lời cùng bằng chứng/nhật ký. Dự án hiện ưu tiên **solve-first**:
planner phải tạo plan khai báo hợp lệ trước khi executor thực hiện; hệ thống
không nhận SQL hay mã thực thi từ câu hỏi.

> **Trạng thái hiện tại:** checkpoint Qwen2.5-0.5B-Instruct đã có tại đường dẫn
local được chỉ định và đã load/inference trên GTX 1060. Smoke test 3 câu tự
tạo sinh ra 0 QueryPlan hợp lệ sau một lần sửa; backend đã abstain an toàn.
Chưa có benchmark accuracy end-to-end hoặc xác minh JSONL submission với
validator hiện hành của cuộc thi. Có thể chọn CUDA hoặc CPU tường minh; backend
không tự fallback.

## 1. Kiến trúc

```text
questions.jsonl
      │
      ▼
Question Understanding (Transformers local-only)
      │ QueryPlan JSON
      ▼
Schema/plan validation → data loader → deterministic executor
      │ answer + evidence + trace
      ├── run log JSONL
      └── submission JSONL (schema tham khảo, chưa được portal xác nhận)
```

- `scripts/run_inference.py`: entrypoint batch inference thống nhất.
- `src/datamate/local_inference.py`: tải checkpoint từ thư mục local, tắt
  truy cập Hugging Face Hub/offline fallback, không gọi API; CPU/CUDA phải được
  chọn tường minh và cấu hình precision tương thích thiết bị.
- `src/datamate/data_loader.py`, `query_plan.py`, `query_executor.py` và
  `pipeline.py`: khám phá/đọc CSV, XLSX, SQLite, kiểm tra QueryPlan và thực thi.
- `src/datamate/business_context.py`: nạp business README, rulebook và calendar
  có giới hạn kích thước.
- `scripts/run_manual_test.py`: chạy câu hỏi người dùng tự viết.
- `scripts/replay_run.py`: replay một QueryPlan đã ghi, không chạy lại model.
- `scripts/evaluate_predictions.py`: evaluator local để chẩn đoán; không phải
  metric/validator chính thức.
- `scripts/prepare_plan_training.py`, `scripts/train_query_plans.py` và
  `scripts/merge_lora_adapter.py`: công cụ chuẩn bị nhãn QueryPlan đã duyệt,
  train và merge adapter. Đây không phải đường inference và chưa được chứng
  minh chạy với checkpoint/data huấn luyện thực tế.

Các business rules chưa có executor tổng quát; tài liệu rulebook được đưa vào
planner context nhưng không đồng nghĩa mọi quy tắc đã được thực thi. Rule tag
không hỗ trợ sẽ bị từ chối, không được đoán.

## 2. Yêu cầu

- Python **3.10–3.12**; khuyến nghị Python 3.12. Đã kiểm thử mã trên Python
  3.12.6 tại môi trường phát triển.
- Dung lượng đĩa cho Python packages, dữ liệu và checkpoint.
- Inference có thể dùng NVIDIA GPU với driver/CUDA tương thích và VRAM đủ, hoặc
  CPU khi `device` được đặt tường minh thành `cpu` cùng precision `fp32`. Không
  tự fallback giữa thiết bị khi thiếu VRAM hoặc gặp lỗi.
- Checkpoint Qwen2.5-0.5B-Instruct đã có sẵn tại đường dẫn local cấu hình riêng;
  dự án không phân phối model/checkpoint và không tự tải model.
- Dữ liệu doanh nghiệp được chuẩn bị cục bộ theo định dạng mà loader hỗ trợ.

Chưa có con số VRAM tối thiểu được xác nhận: nó phụ thuộc kích thước checkpoint,
precision, độ dài prompt và dữ liệu/cấu hình thực tế. Không nên coi tên model
ứng viên trong planning là model đã được hỗ trợ hoặc benchmark.

## 3. Cài đặt

Xem [INSTALL.md](INSTALL.md) để cài từng bước trên Windows và Ubuntu/Linux,
hoặc [MODEL_SETUP.md](MODEL_SETUP.md) để nối checkpoint local vào inference.
Inference dependencies ở `requirements-inference.txt`; PyTorch phải khớp GPU,
driver và môi trường của máy. `requirements-dev.txt` chỉ cài dependency nhẹ để
phát triển/test, không cài PyTorch hoặc Transformers.

## 4. Chuẩn bị model

Chọn một checkpoint được phép sử dụng theo giấy phép và quy định cuộc thi, rồi
chuẩn bị đầy đủ tokenizer/config/weights trong một thư mục local. DataMate chỉ
đọc đường dẫn thư mục này với `local_files_only=True` và
`trust_remote_code=False`; checkpoint thiếu hoặc cần custom remote code sẽ bị
từ chối. LoRA adapter cần được merge/export thành model local tương thích trước
khi dùng backend này.

Ví dụ đường dẫn:

```text
models/
└── local-checkpoint/
    ├── config.json
    ├── tokenizer files...
    └── model weight files...
```

Đây chỉ là layout minh họa, không phải checkpoint có sẵn trong repository.

## 5. Chuẩn bị dữ liệu

`business_data_root` chứa một thư mục cho mỗi `business_id`. Trong mỗi thư mục,
đặt các CSV, XLSX hoặc SQLite cần truy vấn và có thể đặt `README.md` mô tả
doanh nghiệp. `rulebook.md` và `calendar.csv` có thể được phát hiện cạnh thư mục
root, hoặc truyền đường dẫn tường minh.

Input là UTF-8 JSONL, một object mỗi dòng:

```json
{"id":"my-question-001","business_id":"biz_example","question":"Tổng doanh thu trong tháng 01/2026 là bao nhiêu?","as_of_date":"2026-02-01"}
```

Các trường bắt buộc là `id`, `business_id`, `question`; `as_of_date` tùy chọn
và phải là ngày ISO-8601. Nếu bỏ qua, có thể đặt ngày mặc định trong config.
Không đưa dữ liệu cá nhân hoặc dữ liệu không được phép vào Git.

## 6. Chạy DataMate

Entry point inference duy nhất:

```text
python scripts/run_inference.py --config config/inference.local.json
```

Tạo file local từ `config/inference.example.json`, sửa đường dẫn theo máy và
đặt checkpoint/dữ liệu/output bên ngoài source nếu có thể. File
`config/inference.local.json` được Git ignore. Các path tương đối trong config
được tính từ thư mục chứa file config. Có thể override cài đặt bằng flags, ví
dụ `--precision fp16`.

Các trường config: `input`, `business_data_root`, `model_path`, `log`,
`submission_output` (bắt buộc); `coercions`, `rulebook`, `calendar`,
`as_of_date`, `device`, `precision`, `max_prompt_tokens`, `max_new_tokens`,
`parameter_cap` (tùy chọn). `local_files_only` phải là `true` và
`trust_remote_code` phải là `false`; backend từ chối cấu hình ngược lại. Xem
config mẫu. Cấu hình riêng cho checkpoint
Qwen có tại `config/inference.qwen.local.example.json`; điều chỉnh input,
data root và output trước khi chạy.

`as_of_date` trong từng câu hỏi được ưu tiên hơn ngày mặc định cấu hình.
Submission được ghi theo schema tham khảo trong training rulebook (`id/type/
answer`) và chỉ xuất bản khi tất cả dòng thành công. Schema này **chưa được
xác minh với validator cuộc thi**. Run log là artifact nội bộ khác submission.
Đường dẫn output không được nằm trong thư mục input hoặc thư mục dữ liệu.

### Câu hỏi do người dùng tự viết

`manual_test/questions.jsonl` để trống cho người dùng tự thêm câu hỏi. Chạy:

```text
python scripts/run_manual_test.py --business-data-root <business-root> --model-path <checkpoint-local>
```

Chi tiết schema kết quả/đáp án chuẩn xem [manual_test/README.md](manual_test/README.md).
Không dùng file manual test làm nguồn train.

## 7. Kết quả

- `submission_output`: JSONL tham khảo để kiểm tra thủ công; chưa được chứng
  nhận là submission chính thức.
- `log`: JSONL nội bộ ghi câu hỏi, model, QueryPlan, trace, evidence, thời gian
  và lỗi.
- CLI in tổng số câu thành công/thất bại. Nếu có câu lỗi, trả exit code khác 0
  và không publish submission một phần; lỗi được ghi ra stderr.

Có thể replay một record log bằng `scripts/replay_run.py`; evaluator local chạy
bằng `scripts/evaluate_predictions.py`. Cả hai chỉ phục vụ chẩn đoán cục bộ.

## 8. Kiểm thử

Tạo môi trường dev rồi chạy từ repo root:

```text
python -m pip install -r requirements-dev.txt
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

Ubuntu/Linux:

```bash
PYTHONPATH=src python -m pytest -q
```

Sau đó chạy:

```text
python -m compileall -q src tests scripts
git diff --check
```

Test integration dùng SQLite/fixture tổng hợp và planner giả lập. Chúng xác
nhận plumbing/executor, không đo khả năng hiểu câu hỏi của model thật. Không cần
public/private test để chạy test suite này.

## 9. Offline và Docker

Sau khi đã cài packages và chuẩn bị checkpoint cục bộ, đường chạy inference đặt
Transformers/Hugging Face ở chế độ offline, không gọi API và không tự tải model.
Việc cài package hay lấy checkpoint trước đó có thể cần mạng; đó không phải
inference offline.

Có cấu hình Docker CPU để kiểm tra plumbing local tại [DOCKER.md](DOCKER.md),
`Dockerfile.dev` và `compose.dev.yaml`. Container chặn network lúc inference,
mount `/data` và checkpoint chỉ đọc, ghi output vào `/result`, và dùng giới hạn
prompt 2.048 tokens với tokenizer thực tế. Đây là cấu hình development, không
phải image nộp cuộc thi.

Quy chế DataMate công khai xác nhận inference offline, `/data` read-only,
`/result/submission.jsonl`, tối đa 15B tham số gộp và giới hạn 6 giờ trên máy
2× RTX 5090. Quy chế cũng dẫn tới Docker guideline riêng để xác định base image
và runtime. Guideline đó chưa truy cập được, nên chưa thể khẳng định image dev
tương thích yêu cầu nộp cuối; cần áp dụng đúng base image/đóng gói model theo
hướng dẫn chính thức trước khi nộp.

## 10. Lỗi thường gặp và giới hạn

- **Thiếu PyTorch/Transformers:** cài inference requirements và bản PyTorch
  đúng CUDA; không cài model tự động.
- **CUDA không khả dụng/không đủ VRAM:** backend không tự fallback; đặt
  `device` thành `cpu` và `precision` thành `fp32` để thử CPU inference.
- **GPU không hỗ trợ bf16:** chạy với `--precision fp16` nếu GPU hỗ trợ; nếu
  không, cần chọn cấu hình phù hợp thay vì giả định.
- **Không tìm thấy bảng/cột:** kiểm tra business root, `business_id`, header
  Excel/CSV và schema thực tế; không ép planner dùng cột không tồn tại.
- **Submission không được cuộc thi chấp nhận:** schema hiện chỉ tham khảo
  training rulebook; cần đối chiếu hướng dẫn/validator chính thức.
- **Kết quả nghiệp vụ sai:** xem QueryPlan/evidence/trace trong run log và replay
  cục bộ; không coi test fixture là bằng chứng accuracy thực tế.

Xem [planning/21_LOCAL_MODEL_INTEGRATION_REPORT.md](planning/21_LOCAL_MODEL_INTEGRATION_REPORT.md)
để biết kết quả smoke test model và các giới hạn hiện tại; báo cáo deployment
trước đó ở [planning/17_DEPLOYMENT_REPORT.md](planning/17_DEPLOYMENT_REPORT.md).
