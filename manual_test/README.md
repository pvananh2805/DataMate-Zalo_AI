# Kiểm thử thủ công DataMate

`questions.jsonl` là file riêng để bạn tự viết câu hỏi. File này được tạo rỗng; không có câu hỏi training, public-test hay private-test được chép vào đây. Mỗi dòng là một JSON object độc lập.

## Cấu trúc câu hỏi

Các trường được hỗ trợ:

- `id`: mã duy nhất do bạn tự đặt.
- `business_id`: tên thư mục doanh nghiệp nằm dưới `--business-data-root`.
- `as_of_date`: ngày tham chiếu dạng `YYYY-MM-DD`, hoặc `null` nếu không cần.
- `question`: câu hỏi nguyên bản gửi cho planner.
- `expected_answer`: `null` nếu chỉ muốn xem dự đoán; nếu có đáp án chuẩn, dùng object `{"type": ..., "answer": ...}`.

Schema `expected_answer` theo kiểu đáp án và value mà evaluator hiện tại đọc từ submission:

```json
{"id":"my-question-1","business_id":"my-business","as_of_date":null,"question":"Câu hỏi bạn tự viết","expected_answer":null}
{"id":"my-question-2","business_id":"my-business","question":"Một câu hỏi tự viết khác","expected_answer":{"type":"number","answer":42}}
```

Hai dòng trên chỉ minh họa cấu trúc, không phải dữ liệu test và không được thêm sẵn vào `questions.jsonl`. Các `type` hợp lệ: `number`, `percent`, `string`, `date`, `boolean`, `list`, `table`, `unanswerable`. `answer` dùng JSON tương ứng với type. Với `table`, theo evaluator cần dạng:

```json
{"type":"table","answer":{"columns":["product","total"],"rows":[["A",42]]}}
```

Với `unanswerable`, đặt `answer` là `null`. Không dùng `abstain` làm đáp án chuẩn. So khớp dùng quy tắc giá trị của `evaluation.py`; không có alias trong định dạng manual này. Accuracy là phép đo local, không phải Penalized Accuracy chính thức.

`expected_answer` chỉ được đọc/kiểm tra để so sánh sau khi pipeline đã tạo dự đoán. Planner chỉ nhận nội dung `question` và context nghiệp vụ; đáp án chuẩn không được đưa vào prompt hoặc executor.

## Chạy

Từ thư mục gốc repository, trên máy có CUDA, dependencies inference và checkpoint local:

```powershell
python scripts\run_manual_test.py `
  --business-data-root "D:\path\to\businesses" `
  --model-path "D:\path\to\local-checkpoint"
```

Mặc định CLI đọc `manual_test/questions.jsonl`, ghi chi tiết vào `manual_test/results.jsonl` và tổng hợp vào `manual_test/report.json`. Có thể truyền `--questions`, `--results`, `--report`, `--rulebook` hoặc `--calendar` để đổi đường dẫn. Không cần bật Internet; model phải có sẵn tại `--model-path`. Backend hiện yêu cầu CUDA, do đó máy CPU-only chưa chạy được inference này.

Mỗi dòng kết quả có `generated_answer`, `expected_answer`, `prediction_status`, `correctness`, `query_plan`, `evidence`, `execution_trace`, `error`, `error_category` và `elapsed_time` (value theo millisecond). Câu lỗi vẫn được ghi ra kết quả và không bị bỏ qua. Accuracy chỉ tính trên `expected_answer` hợp lệ; `abstain` không được tính đúng và được đếm riêng như evaluator (không tính vào số câu sai trong thống kê reward local).

Với mọi câu trả lời sai, hệ thống không thể suy ra nguyên nhân chỉ từ đáp án cuối; `error_category` là `Cần phân tích thủ công`, `error_category_candidates` liệt kê nhóm cần xem xét và `answer_type_matches` cho biết type có khớp không. Đối chiếu `query_plan`, `evidence` và `execution_trace`, rồi phân loại nguyên nhân theo một trong các nhóm: hiểu câu hỏi; QueryPlan; chọn bảng/cột; lọc; thực thi; tính toán; business rule; định dạng; hoặc dữ liệu không đủ. Không sửa đáp án chuẩn để làm cho prediction khớp.

## Dữ liệu riêng tư

`questions.jsonl`, `results.jsonl`, `report.json` và các file khác trong thư mục này (ngoại trừ README) bị bỏ qua cục bộ bằng `.git/info/exclude`; không stage hoặc push dữ liệu kiểm thử cá nhân.
