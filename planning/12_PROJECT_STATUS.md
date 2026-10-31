# Trạng thái dự án DataMate

**Cập nhật:** 2026-10-31
**Mốc nguồn đã xác minh:** `70976ca` (loader ban đầu; đã đẩy lên repository riêng trước đó).

## Đã triển khai và kiểm chứng

- Đã tải, kiểm tra CRC/đường dẫn và giải nén archive training riêng tại `data\train\train`. Training có 3.000 cặp câu hỏi/đáp án, 30 doanh nghiệp, 20 CSV, 107 XLSX/302 worksheet và 1 SQLite/17 bảng.
- Đã mở rộng `src/datamate/data_loader.py`: đọc CSV theo luồng (UTF-8/BOM, UTF-16/32 BOM, CP1258, CP1252; tự nhận diện delimiter), XLSX read-only và SQLite read-only; đường dẫn được giới hạn trong thư mục business. Có API stream nhiều sheet trong cùng workbook mà không mở workbook lại cho từng sheet.
- Với 81 worksheet có dòng tiêu đề/metadata trước bảng, loader chọn header sau dòng trống theo heuristic và lưu `header_row`/cảnh báo; không xóa hoặc sửa file gốc. Header trùng, header trống và dòng dữ liệu rộng hơn vẫn được báo rõ để kiểm tra.
- Đã chạy full streaming profile: **30 doanh nghiệp, 339 bảng/worksheet, 3.997.076 dòng, 0 lỗi đọc**. Báo cáo: `planning/15_SCHEMA_PROFILING.md`; JSON chi tiết 2.026.467 byte tại `data\train\schema_profile.json` (artifact local, không đưa vào commit).
- Profile ghi kiểu quan sát, null, nhãn thời gian/tiền/số lượng/trạng thái/định danh, khóa ứng viên và 227 quan hệ gợi ý theo tên cột. Khóa/quan hệ CSV-XLSX đều là heuristic, chưa được xác nhận để join tự động.
- Đã thống kê đủ 3.000 nhãn đáp án: number 1.884; percent 578; list 193; date 145; string 88; unanswerable 70; boolean 23; table 19. ID câu hỏi/đáp án khớp đầy đủ; không có record JSONL lỗi.
- Đã thêm QueryPlan có cấu trúc và kiểm tra bảng/cột, join liên thông, filter, aggregation, sort, time range, limit, output type và evidence reference; không nhận SQL hay mã thực thi từ câu hỏi.
- Đã thêm executor xác định, streaming theo hàng, có join giới hạn bộ nhớ, filter/group, SUM/COUNT/AVG/MIN/MAX, ranking/sort/top-K, tỷ lệ/tăng trưởng, trace và tổng Decimal. Executor yêu cầu đầu vào có kiểu tương thích; hiện chưa có bước tự chuẩn hóa chuỗi CSV tiền/số/ngày thành giá trị typed.
- Đã thêm kiểm tra kiểu câu trả lời, finite number/percent, date, boolean, list/table, evidence và các trạng thái no-data/unanswerable/ambiguous/insufficient-data/abstain. Pipeline `execute_business_plan` nối loader → plan validation → executor → answer validation trên fixture SQLite.
- Đã nối logging JSONL cho pipeline: lưu input, model, QueryPlan, trace, evidence, answer, thời gian và lỗi đã nhận diện; có đọc lại/kiểm schema log và chặn ghi trong input root. Chưa có chức năng tái thực thi từ log.
- `pytest -q`: **56 passed** ở lần kiểm tra gần nhất; `compileall` cho `src`, `tests`, `scripts` thành công.
- Public-test archive vẫn nguyên trạng, chưa giải nén/đọc. Không có dữ liệu training/public-test trong source commit trước đó.

## Đã triển khai nhưng chưa đủ để dùng trong inference cuộc thi

- Header Excel được phát hiện heuristic; 81 worksheet có dòng metadata/tiêu đề đầu trang bị bỏ qua có cảnh báo. Cần xác nhận từng header trước khi planner tin cậy.
- Pipeline executor mới chạy khi có QueryPlan hợp lệ. Chưa có bộ hiểu câu hỏi tiếng Việt/model adapter tạo plan từ câu hỏi tự nhiên.
- Answer validator và executor có unit/integration tests; chưa có serializer chính thức vì chưa xác minh schema output/evaluator của ban tổ chức.
- Evidence hiện là tham chiếu bảng/cột, số dòng và filter summary; chưa có hệ thống trích dẫn record-level đủ để tái kiểm chứng mọi phép tính.

## Chưa triển khai

| Hạng mục | Trạng thái / việc còn lại |
|---|---|
| Question understanding tiếng Việt | Chưa có parser/model inference tích hợp; cần ánh xạ ý định, thực thể, từ đồng nghĩa, thời gian và điều kiện vào QueryPlan. |
| Chuẩn hóa dữ liệu | Chưa có bộ chuyển đổi có kiểm soát cho chuỗi CSV/XLSX sang Decimal/ngày/boolean; cần phát hiện định dạng mơ hồ và từ chối thay vì đoán. |
| Query planning/executor hoàn chỉnh | Plan validator và executor hiện có là lõi thử nghiệm; rulebook chưa được thực thi trong executor, chưa hỗ trợ so sánh hai kỳ và chứng cứ đầu ra chưa đủ để tái kiểm chứng mọi phép tính. |
| Model selection/offline inference | Chưa chọn/tải model; chưa benchmark parameter count, ngày phát hành, license, tiếng Việt, tốc độ/RAM và cấu trúc plan. |
| Training/fine-tuning | Chưa có pipeline dữ liệu plan đã gán nhãn, split chống leakage, training/LoRA, validation/export/checkpoint. Máy hiện tại không có GPU NVIDIA; không chạy fine-tuning lớn trên CPU. |
| Evaluation | Chưa chạy metric chính thức. Chưa có nguồn evaluator/công thức PA chính thức; không công bố điểm PA. |
| Logging/reproducibility | Đã có JSONL log input/plan/trace/evidence/answer/lỗi; chưa có chức năng tái chạy end-to-end từ log, lưu phiên bản dependency và checkpoint. |
| Docker/submission | Docker CLI chưa có; chưa xác minh Docker guideline, entrypoint, đường dẫn và schema submission. |
| Tài liệu cuộc thi | Portal xác nhận mô tả cấp cao và `.jsonl`; regulations chỉ trả SPA shell. Giới hạn 15B, ngày phát hành, license, quy tắc offline/dữ liệu, PA và Docker vẫn cần văn bản chính thức để xác minh. |

## Dữ liệu, tài nguyên và an toàn

- Training ZIP: 128.118.095 byte; SHA-256 `C1ADBC654A301C30E4CF20B55D301381025F49DB1162D88322B552D7472FEA01`.
- Public-test ZIP: 64.149.714 byte; SHA-256 `46FB50F926ED1510B9F099A53918F45E304B256680452C93E74401B4A5442C52`; archive chưa được mở.
- Máy phát triển: Intel i3-1115G4, RAM 12 GB, Intel UHD, không có NVIDIA/CUDA; Docker CLI không có. Trong lần profile streaming, working set quan sát gần cuối khoảng 45 MB (không phải số đo peak).
- Dữ liệu gốc chỉ đọc; source profile JSON và dataset nằm ngoài phạm vi commit. Không sử dụng public/private test để huấn luyện hoặc ghi nhớ đáp án.

## Mốc tiếp theo

1. Hoàn thiện bộ chuẩn hóa giá trị có kiểu và tests trên các CSV/XLSX bất thường.
2. Phát triển question-understanding/model adapter độc lập, chỉ xuất QueryPlan đã validate.
3. Hoàn thiện evidence/answer/output/logging và evaluation trên training validation khi có schema/metric được phép.
4. Chuẩn bị training/fine-tuning có thể chạy trên GPU khác; không huấn luyện lớn trên laptop.
5. Chỉ chốt model, Docker và submission sau khi xác minh tài liệu chính thức.

## Xác nhận phạm vi

Đây là tiến độ thực tế, không phải tuyên bố DataMate hoàn chỉnh. Chưa có model inference, training pipeline đã chạy, official evaluation, public-test run, Docker validation hay submission cuối cùng. Không tạo hoặc ghi nhận điểm thi chưa đo.
