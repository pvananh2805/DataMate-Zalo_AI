# Báo cáo Solve-First — đánh giá năng lực truy vấn

## Kết luận ngắn

Chưa có cơ sở đo độ chính xác end-to-end hoặc tuyên bố DataMate đã giải được một dạng câu hỏi thực tế. Bộ dữ liệu train có 3.000 câu hỏi và 3.000 đáp án, nhưng không có dự đoán của một model/planner đang chạy để đối chiếu; đáp án gold không phải nhãn QueryPlan. Bộ test chứng minh phép toán trên fixture và pipeline executor, không chứng minh planner hiểu và ánh xạ đúng câu hỏi train.

Trong vòng này, đã sửa lỗi lọc ngày khiến giá trị `date`/`datetime` đã chuẩn hóa không so sánh được với mốc ISO dạng chuỗi; đồng thời loại bỏ lọc từ vựng không an toàn cho ngày sai định dạng. QueryPlan/executor hiện cho phép mỗi aggregation có filter và time range riêng để đối chiếu hai kỳ trên cùng bảng, và làm tròn kết quả phần trăm đến 2 chữ số với quy tắc nửa làm tròn lên theo rulebook training. Public test không được mở. Không commit hoặc push.

## Thống kê dữ liệu train

3.000 đáp án gold có loại: `number` 1.884; `percent` 578; `list` 193; `date` 145; `string` 88; `unanswerable` 70; `boolean` 23; `table` 19. ID của câu hỏi và đáp án khớp đầy đủ. 70 nhãn `unanswerable` là số liệu trực tiếp từ answers, không phải dự đoán của hệ thống.

Đếm từ khóa trên câu hỏi cho thấy các tín hiệu sau (các nhóm **chồng lấp**, không phải taxonomy hoặc nhãn chuẩn): từ ngữ tổng hợp/doanh thu 972; kỳ/thời gian 1.572; phần trăm/tỷ lệ 557; lọc/đối tượng 216; trung bình 173; đếm 179; xếp hạng/từ “nhất” 331; min/max 112; so sánh kỳ/tăng giảm 182; cách nói phân nhóm theo tháng/ngày/nhóm 12. Regex không nhận diện đủ cách diễn đạt; các số này chỉ dùng để định hướng ưu tiên, không được xem là tần suất thật của phép toán.

Các câu hỏi train được đọc có dạng bao gồm tổng phiếu nhập/tháng, tổng doanh thu thuốc theo hoạt chất, đếm đơn hủy, trung bình giá nhập theo số lượng, top mặt hàng bán chạy, ngày đạt doanh thu cao nhất, báo doanh thu từng tháng trong ba tháng liên tiếp và phần trăm thay đổi số lượng bán giữa hai cửa sổ bảy ngày. Điều này cho thấy cần xử lý thời gian, business rules và phân nhóm; chỉ có `SUM`/`GROUP BY` cơ bản chưa đủ.

## Đối chiếu năng lực hiện tại

| Dạng | Trạng thái dựa trên bằng chứng hiện có |
|---|---|
| `SUM`, `COUNT`, `AVG`, `MIN`, `MAX`, `NULL` | Các primitive có test trên fixture; chưa benchmark ánh xạ câu hỏi thật hoặc độ chính xác đầu-cuối. |
| Filter và khoảng thời gian theo ngày | Executor có test; lỗi `date`/`datetime` được sửa trong vòng này. Chưa xác minh policy/coercion trên mọi bảng train. |
| `GROUP BY`, `JOIN`, sort, top-K | Có test executor; top-K kiểm tra được khi plan nêu rõ tiêu chí phụ tie-break. Chưa xác minh model tạo đúng plan, khóa join và sort theo quy tắc mã/tên. |
| Tỷ lệ, chênh lệch, phần trăm thay đổi | Có phép tính trên aggregate alias; đã thêm period scope riêng theo từng aggregation, kiểm thử so sánh hai kỳ từ cùng bảng qua pipeline trên fixture; phần trăm làm tròn 2 chữ số nửa lên. Chưa benchmark planner hoặc trả lời câu hỏi train; kỳ không có dòng vẫn cần phân biệt với kỳ nằm ngoài vùng dữ liệu. |
| Phân nhóm theo tháng/tuần/quý; nhiều kỳ liên tiếp | Chưa có time-bucket/grouping tổng quát trong QueryPlan/executor; chưa giải được câu hỏi “doanh thu mỗi tháng trong ba tháng” từ dữ liệu chi tiết chỉ có timestamp/ngày. |
| Rulebook doanh thu, trả hàng, công nợ, tồn kho, đơn vị và luật ngành | Chưa được executor thực thi tổng quát. Rule tag chưa hỗ trợ bị từ chối; không được tuyên bố những dạng cần các luật này đã giải. |
| `unanswerable`, mơ hồ, thiếu dữ liệu, abstain | Có đường biểu diễn/validation; chưa chạy trên 70 ví dụ gold và chưa đánh giá phân biệt đúng giữa không có dữ liệu, kỳ ngoài vùng dữ liệu và thiếu trường. |

**Độ chính xác hiện có:** chưa đo được (`N/A`), không phải 0%. Không có prediction hợp lệ từ model/planner để so với 3.000 gold. Evaluator local chỉ là công cụ chẩn đoán; tài liệu hiện có chưa xác minh công thức Penalized Accuracy tổng hợp hoặc validator cuộc thi.

## Lỗi đã sửa và kiểm chứng

- **Nguyên nhân:** `_time_matches` so mốc `str` trong QueryPlan trực tiếp với `date`/`datetime` sau normalization, gây `TypeError`. Chuỗi timestamp cũng bị so từ vựng với ngày cuối kỳ, có thể loại nhầm các phát sinh trong ngày cuối.
- **Sửa:** so theo ngày lịch; hỗ trợ `date`, `datetime`, chuỗi ISO date/datetime; ngày cuối được tính trọn khi `inclusive_end=True` và bị loại khi là mốc kết thúc độc quyền. Chuỗi ngày sai định dạng bị báo lỗi thay vì cho kết quả lọc từ vựng.
- **Lỗi precision:** `ratio_percent` và `percent_change` trả về số thập phân tuần hoàn nhiều chữ số, trái với rulebook yêu cầu phần trăm 2 chữ số. Đã làm tròn bằng `ROUND_HALF_UP`, giữ độ chính xác cao trước khi lượng tử hóa và tăng precision cho giá trị lớn.
- **Test hồi quy:** date và datetime đã chuẩn hóa, timestamp cuối ngày, cận kết thúc độc quyền, ISO text và giá trị ngày sai định dạng; thêm kiểm tra phép giảm phần trăm âm, min/max/top-K tie-break, period scope trên hai kỳ và integration pipeline/evidence.
- **Kết quả:** `pytest -q` toàn bộ **128 passed**; `compileall` và `git diff --check` thành công.

## Việc còn yếu / blocker

1. Chưa có model checkpoint phù hợp, prediction trên tập đánh giá hoặc annotation QueryPlan; vì vậy chưa thể đo accuracy hay khẳng định năng lực question-to-plan.
2. Cần time-bucket/grouping theo tháng/tuần/quý; các câu hỏi train yêu cầu doanh thu từng tháng trong nhiều kỳ chưa được giải bằng executor.
3. Cần thực thi từng business rule có dữ liệu và test đối chứng; hiện rulebook chủ yếu được đưa vào prompt nhưng các luật chưa biểu diễn được vẫn bị fail-closed.
4. Cần bộ benchmark/annotation có thể kiểm chứng cho 70 trường hợp unanswerable và câu hỏi mơ hồ/thiếu trường; không suy ra cách giải từ answer gold.
5. Chưa xác minh được evaluator PA/validator challenge-specific và yêu cầu submission; chưa có GPU/model để inference benchmark hoặc fine-tune. Public test chưa được mở và không được dùng.

**Phạm vi vòng này:** sửa temporal filter, thêm period scope cho từng aggregation, áp dụng precision phần trăm theo rulebook và bổ sung test đơn vị/integration. Không fine-tune, không mở public test, không tạo nhãn giả, không commit/push. Kết quả này là một bước năng lực executor đã kiểm chứng, chưa phải milestone giải câu hỏi end-to-end.
