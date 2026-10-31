# Trạng thái dự án DataMate

**Cập nhật:** 2026-10-31
**Mốc code đã xác minh trước thay đổi hiện tại:** `dda9621` (schema profiling và lõi truy vấn xác định; đã đẩy lên repository riêng).

## Đã triển khai và kiểm chứng

- Đã tải, kiểm tra CRC/đường dẫn và giải nén archive training riêng tại `data\train\train`. Training có 3.000 cặp câu hỏi/đáp án, 30 doanh nghiệp, 20 CSV, 107 XLSX/302 worksheet và 1 SQLite/17 bảng.
- Đã mở rộng `src/datamate/data_loader.py`: đọc CSV theo luồng (UTF-8/BOM, UTF-16/32 BOM, CP1258, CP1252; tự nhận diện delimiter), XLSX read-only và SQLite read-only; đường dẫn được giới hạn trong thư mục business. Có API stream nhiều sheet trong cùng workbook mà không mở workbook lại cho từng sheet.
- Với 81 worksheet có dòng tiêu đề/metadata trước bảng, loader chọn header sau dòng trống theo heuristic và lưu `header_row`/cảnh báo; không xóa hoặc sửa file gốc. Header trùng, header trống và dòng dữ liệu rộng hơn vẫn được báo rõ để kiểm tra.
- Đã chạy full streaming profile: **30 doanh nghiệp, 339 bảng/worksheet, 3.997.076 dòng, 0 lỗi đọc**. Báo cáo: `planning/15_SCHEMA_PROFILING.md`; JSON chi tiết 2.026.467 byte tại `data\train\schema_profile.json` (artifact local, không đưa vào commit).
- Profile ghi kiểu quan sát, null, nhãn thời gian/tiền/số lượng/trạng thái/định danh, khóa ứng viên và 227 quan hệ gợi ý theo tên cột. Khóa/quan hệ CSV-XLSX đều là heuristic, chưa được xác nhận để join tự động.
- Đã thống kê đủ 3.000 nhãn đáp án: number 1.884; percent 578; list 193; date 145; string 88; unanswerable 70; boolean 23; table 19. ID câu hỏi/đáp án khớp đầy đủ; không có record JSONL lỗi.
- Đã thêm QueryPlan có cấu trúc và kiểm tra bảng/cột, join liên thông, filter, aggregation, sort, time range, limit, output type và evidence reference; không nhận SQL hay mã thực thi từ câu hỏi.
- Đã thêm executor xác định, streaming theo hàng, có join giới hạn bộ nhớ, filter/group, SUM/COUNT/AVG/MIN/MAX, ranking/sort/top-K, tỷ lệ/tăng trưởng, trace và tổng Decimal.
- Đã thêm `src/datamate/normalization.py`: chuẩn hóa theo policy tường minh cho number/integer/date/datetime/boolean/text/identifier; không đoán locale/ngày, giữ streaming, từ chối giá trị sai/row-width/header trùng. Pipeline nhận coercion theo `ColumnRef` và ghi policy vào metadata log để tái lập; chưa tự sinh policy từ profile.
- Đã thêm bộ giải mã JSON nghiêm ngặt cho QueryPlan và adapter question-understanding độc lập model: prompt có schema/ngày tham chiếu/rulebook context, từ chối JSON lỗi, field lạ, key trùng, non-finite, code fence và cột ngoài schema. Pipeline tích hợp đường đi câu hỏi → planner → executor → validator → log; adapter chỉ nhận backend được inject, chưa chọn hoặc tải model inference.
- Đã thêm `src/datamate/local_inference.py` và `scripts/run_inference.py`: backend Transformers chỉ tải checkpoint từ thư mục local, `local_files_only`, không bật remote code/API, generation greedy xác định, kiểm trần tham số và từ chối chạy CPU. Entrypoint xử lý câu hỏi JSONL tuần tự, nhận coercion config được duyệt và ghi run log nội bộ; không xuất submission vì schema chính thức chưa xác minh.
- Đã thêm `scripts/prepare_plan_training.py` và `src/datamate/training_data.py`: nhận nhãn QueryPlan do con người duyệt, kiểm tra theo schema thật, tạo prompt giống inference và chia train/validation theo `business_id` bằng seed cố định. JSONL được đọc/ghi theo luồng, staging bằng SQLite trên đĩa, từ chối output đè dữ liệu hiện có; không suy ra plan từ answer và không đọc public-test. Hiện chưa có bộ annotation plan để tạo training set thực tế.
- Đã chuẩn bị `scripts/train_query_plans.py`, `src/datamate/training_runner.py`, `requirements-training.txt` và `scripts/merge_lora_adapter.py`: chạy offline trên CUDA, hỗ trợ full fine-tune hoặc LoRA với target modules do người dùng khai báo, từ chối CPU/thiếu CUDA, kiểm tra leakage và tổng parameter sau adapter, lưu checkpoint/log/eval loss và có bước merge/export LoRA. Chưa cài dependency training, chưa chạy vì máy không có GPU/model/annotation; QLoRA không được giả định hỗ trợ.
- Đã thêm kiểm tra kiểu câu trả lời, finite number/percent, date, boolean, list/table, evidence và các trạng thái no-data/unanswerable/ambiguous/insufficient-data/abstain. Pipeline `execute_business_plan` nối loader → plan validation → executor → answer validation trên fixture SQLite.
- Nhánh question-to-plan đã có đường abstain/unanswerable rõ ràng: `ambiguous`, `insufficient_data`, `unanswerable` hoặc `abstained` được log với lý do và không giả lập execution/evidence; `no_data` chỉ được gán sau khi executor thực sự chạy và không có dòng khớp.
- Đã nối logging JSONL cho pipeline: lưu input, model, QueryPlan, trace, evidence, answer, thời gian và lỗi đã nhận diện; có đọc lại/kiểm schema log và chặn ghi trong input root. Chưa có chức năng tái thực thi từ log.
- Đã thêm `scripts/evaluate_predictions.py` và `src/datamate/evaluation.py`: exact-match chặt theo answer type/value, breakdown theo loại/business, missing/extra/invalid, unanswerable và latency; xử lý Decimal trong list/table từ run log. Report tuyên bố rõ `official_pa_computed=false`, không dùng alias và không tự nhận là PA.
- `PYTHONPATH=src .venv\Scripts\python.exe -m pytest -q`: **82 passed** sau khi tích hợp normalization, question-to-plan/abstain, training-data preparation, evaluator cục bộ và kiểm tra backend offline fail-closed; `compileall` cho `src`, `tests`, `scripts`, CLI trợ giúp chuẩn bị/train/export/inference/evaluation và `git diff --check` thành công. Chưa load/training model thật do không có GPU/checkpoint hợp lệ.
- Public-test archive vẫn nguyên trạng, chưa giải nén/đọc. Không có dữ liệu training/public-test trong source commit trước đó.

## Đã triển khai nhưng chưa đủ để dùng trong inference cuộc thi

- Header Excel được phát hiện heuristic; 81 worksheet có dòng metadata/tiêu đề đầu trang bị bỏ qua có cảnh báo. Cần xác nhận từng header trước khi planner tin cậy.
- Adapter question-understanding hiện chuyển yêu cầu qua backend được inject thành QueryPlan JSON; đây chưa phải model tiếng Việt chạy offline được chọn/đóng gói, và chưa đánh giá độ chính xác ngữ nghĩa trên tập validation.
- Answer validator và executor có unit/integration tests; chưa có serializer chính thức vì chưa xác minh schema output/evaluator của ban tổ chức.
- Evidence hiện là tham chiếu bảng/cột, số dòng và filter summary; chưa có hệ thống trích dẫn record-level đủ để tái kiểm chứng mọi phép tính.

## Chưa triển khai

| Hạng mục | Trạng thái / việc còn lại |
|---|---|
| Question understanding tiếng Việt | Đã có adapter planner, backend Transformers local-only và luồng ambiguity/abstain; chưa chọn/kiểm chứng model đủ điều kiện, chưa chạy checkpoint thật hoặc benchmark ngữ nghĩa. |
| Chuẩn hóa dữ liệu | Đã có coercion có kiểm soát nhưng chưa có cấu hình policy được xác nhận cho từng bảng/cột training; cần profile-assisted policy authoring, kiểm tra dữ liệu đầy đủ và xử lý header Excel cần review. |
| Query planning/executor hoàn chỉnh | Plan validator và executor hiện có là lõi thử nghiệm; rulebook chưa được thực thi trong executor, chưa hỗ trợ so sánh hai kỳ và chứng cứ đầu ra chưa đủ để tái kiểm chứng mọi phép tính. |
| Model selection/offline inference | Backend và CLI offline đã chuẩn bị nhưng chưa tải model; chưa benchmark parameter count thực, ngày phát hành, license, tiếng Việt, tốc độ/RAM và chất lượng QueryPlan. |
| Training/fine-tuning | Đã có pipeline annotation thủ công/group split, script offline GPU full/LoRA, validation theo eval loss và merge/export; chưa có annotation thực, model/checkpoint được chọn, dependency GPU, lần chạy training hoặc đánh giá chất lượng sau fine-tune. QLoRA chưa triển khai. |
| Evaluation | Đã có evaluator exact-match nội bộ và fixture tests; chưa chạy trên model/validation thực, chưa có nguồn evaluator/công thức PA chính thức và không công bố điểm PA. |
| Logging/reproducibility | Đã có JSONL log input/plan/trace/evidence/answer/lỗi; chưa có chức năng tái chạy end-to-end từ log, lưu phiên bản dependency và checkpoint. |
| Docker/submission | Docker CLI chưa có; chưa xác minh Docker guideline, entrypoint, đường dẫn và schema submission. |
| Tài liệu cuộc thi | Portal xác nhận mô tả cấp cao và `.jsonl`; regulations chỉ trả SPA shell. Giới hạn 15B, ngày phát hành, license, quy tắc offline/dữ liệu, PA và Docker vẫn cần văn bản chính thức để xác minh. |

## Dữ liệu, tài nguyên và an toàn

- Training ZIP: 128.118.095 byte; SHA-256 `C1ADBC654A301C30E4CF20B55D301381025F49DB1162D88322B552D7472FEA01`.
- Public-test ZIP: 64.149.714 byte; SHA-256 `46FB50F926ED1510B9F099A53918F45E304B256680452C93E74401B4A5442C52`; archive chưa được mở.
- Máy phát triển: Intel i3-1115G4, RAM 12 GB, Intel UHD, không có NVIDIA/CUDA; Docker CLI không có. Trong lần profile streaming, working set quan sát gần cuối khoảng 45 MB (không phải số đo peak).
- Dữ liệu gốc chỉ đọc; source profile JSON và dataset nằm ngoài phạm vi commit. Không sử dụng public/private test để huấn luyện hoặc ghi nhớ đáp án.

## Mốc tiếp theo

1. Tạo/kiểm chứng policy chuẩn hóa theo schema, tập trung CSV/XLSX và định dạng cột bất thường.
2. Xác minh điều lệ chính thức, chọn checkpoint hợp lệ, chạy backend local trên GPU phù hợp và benchmark trên validation.
3. Hoàn thiện evidence/answer/output/logging; chạy evaluator exact-match trên validation được phép khi đã có model và labels, chỉ thêm PA sau khi nhận công thức chính thức.
4. Thu thập/duyệt nhãn QueryPlan hợp lệ, xác minh model đủ điều kiện và chạy training/export trên GPU phù hợp; không huấn luyện lớn trên laptop.
5. Chỉ chốt model, Docker và submission sau khi xác minh tài liệu chính thức.

## Xác nhận phạm vi

Đây là tiến độ thực tế, không phải tuyên bố DataMate hoàn chỉnh. Chưa có model inference, training pipeline đã chạy, official evaluation, public-test run, Docker validation hay submission cuối cùng. Không tạo hoặc ghi nhận điểm thi chưa đo.
