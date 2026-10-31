# Báo cáo Solve-First — đánh giá năng lực truy vấn

## Kết luận ngắn

Chưa có cơ sở đo độ chính xác end-to-end hoặc tuyên bố DataMate đã giải được một dạng câu hỏi thực tế. Bộ dữ liệu train có 3.000 câu hỏi và 3.000 đáp án, nhưng không có dự đoán của một model/planner đang chạy để đối chiếu; đáp án gold không phải nhãn QueryPlan. Bộ test chứng minh phép toán trên fixture và pipeline executor, không chứng minh planner hiểu và ánh xạ đúng câu hỏi train.

Trong vòng này, đã sửa lỗi lọc ngày khiến giá trị `date`/`datetime` đã chuẩn hóa không so sánh được với mốc ISO dạng chuỗi; đồng thời loại bỏ lọc từ vựng không an toàn cho ngày sai định dạng. QueryPlan/executor hỗ trợ period scope riêng cho từng aggregation và group theo ngày/tuần/tháng/quý/năm; danh sách có thể trả về chuỗi số liệu theo kỳ, bảng có cột kỳ và evidence. Với một time bucket, một aggregation SUM/COUNT, không có chiều group khác và time range cùng cột bucket, pipeline sinh các bucket trống nằm trong giao giữa time range và độ phủ dữ liệu (SUM/COUNT bằng 0); không tự sinh kỳ ngoài độ phủ. Pipeline phân biệt time range hoàn toàn ngoài dữ liệu (unanswerable) với time range nằm trong/chéo vùng dữ liệu. Kết quả phần trăm được làm tròn đến 2 chữ số nửa lên theo rulebook training. Public test không được mở.

## Thống kê dữ liệu train

3.000 đáp án gold có loại: `number` 1.884; `percent` 578; `list` 193; `date` 145; `string` 88; `unanswerable` 70; `boolean` 23; `table` 19. ID của câu hỏi và đáp án khớp đầy đủ. 70 nhãn `unanswerable` là số liệu trực tiếp từ answers, không phải dự đoán của hệ thống.

Đếm từ khóa trên câu hỏi cho thấy các tín hiệu sau (các nhóm **chồng lấp**, không phải taxonomy hoặc nhãn chuẩn): từ ngữ tổng hợp/doanh thu 972; kỳ/thời gian 1.572; phần trăm/tỷ lệ 557; lọc/đối tượng 216; trung bình 173; đếm 179; xếp hạng/từ “nhất” 331; min/max 112; so sánh kỳ/tăng giảm 182; cách nói phân nhóm theo tháng/ngày/nhóm 12. Regex không nhận diện đủ cách diễn đạt; các số này chỉ dùng để định hướng ưu tiên, không được xem là tần suất thật của phép toán.

Các câu hỏi train được đọc có dạng bao gồm tổng phiếu nhập/tháng, tổng doanh thu thuốc theo hoạt chất, đếm đơn hủy, trung bình giá nhập theo số lượng, top mặt hàng bán chạy, ngày đạt doanh thu cao nhất, báo doanh thu từng tháng trong ba tháng liên tiếp và phần trăm thay đổi số lượng bán giữa hai cửa sổ bảy ngày. Điều này cho thấy cần xử lý thời gian, business rules và phân nhóm; chỉ có `SUM`/`GROUP BY` cơ bản chưa đủ.

## Đối chiếu năng lực hiện tại

| Dạng | Trạng thái dựa trên bằng chứng hiện có |
|---|---|
| `SUM`, `COUNT`, `COUNT DISTINCT`, `AVG`, `MIN`, `MAX`, `NULL` | Các primitive có test trên fixture; `count_distinct` bỏ NULL, đếm khóa lặp một lần và giới hạn 100.000 giá trị phân biệt mặc định (vượt giới hạn báo lỗi). Hữu ích cho câu hỏi đếm số đơn/đối tượng trong bảng nhiều dòng trên mỗi đơn. Chưa benchmark ánh xạ câu hỏi thật hoặc độ chính xác đầu-cuối. |
| Filter và khoảng thời gian theo ngày | Executor/pipeline có test; lỗi `date`/`datetime` được sửa. Kiểm tra ngày đầu/cuối độ phủ để từ chối khoảng hoàn toàn ngoài vùng dữ liệu; kiểm thử khoảng chéo vùng và trường hợp SUM bằng 0 trong kỳ có độ phủ. Chưa xác minh policy/coercion trên mọi bảng train. |
| `GROUP BY`, `JOIN`, sort, top-K | Có test executor; top-K kiểm tra được khi plan nêu rõ tiêu chí phụ tie-break. Chưa xác minh model tạo đúng plan, khóa join và sort theo quy tắc mã/tên. |
| Tỷ lệ, chênh lệch, phần trăm thay đổi | Có phép tính trên aggregate alias; đã thêm period scope riêng theo từng aggregation, kiểm thử so sánh hai kỳ từ cùng bảng qua pipeline trên fixture; phần trăm làm tròn 2 chữ số nửa lên. Chưa benchmark planner hoặc trả lời câu hỏi train; kỳ không có dòng vẫn cần phân biệt với kỳ nằm ngoài vùng dữ liệu. |
| Phân nhóm theo ngày/tuần/tháng/quý/năm | QueryPlan/executor và pipeline có time bucket theo lịch; tuần bắt đầu thứ Hai. Integration test xác nhận table JSONL và list chuỗi doanh thu tháng được sort theo kỳ. Executor lấp bucket trống bằng 0 cho một bucket, một hoặc nhiều SUM/COUNT/COUNT DISTINCT không có aggregation time scope, không group thêm, cùng cột time range/bucket; chỉ trong phần giao time range/độ phủ, không sinh kỳ ngoài độ phủ. Nhiều chiều group chưa được lấp bucket trống; chưa benchmark trên câu hỏi train. |
| Rulebook doanh thu, trả hàng, công nợ, tồn kho, đơn vị và luật ngành | Chưa được executor thực thi tổng quát. Rule tag chưa hỗ trợ bị từ chối; không được tuyên bố những dạng cần các luật này đã giải. |
| `unanswerable`, mơ hồ, thiếu dữ liệu, abstain | Đường biểu diễn/validation có sẵn; pipeline đã kiểm thử kỳ hoàn toàn ngoài độ phủ ngày so với kỳ có độ phủ nhưng không có dòng thỏa filter. Chưa đánh giá mô hình phân loại 70 nhãn unanswerable, mơ hồ hoặc thiếu trường; chưa suy ra unanswerable chỉ từ gold answer. |

**Độ chính xác hiện có:** chưa đo được (`N/A`), không phải 0%. Không có prediction hợp lệ từ model/planner để so với 3.000 gold. Evaluator local chỉ là công cụ chẩn đoán; tài liệu hiện có chưa xác minh công thức Penalized Accuracy tổng hợp hoặc validator cuộc thi.

## Lỗi đã sửa và kiểm chứng

- **Nguyên nhân:** `_time_matches` so mốc `str` trong QueryPlan trực tiếp với `date`/`datetime` sau normalization, gây `TypeError`. Chuỗi timestamp cũng bị so từ vựng với ngày cuối kỳ, có thể loại nhầm các phát sinh trong ngày cuối.
- **Sửa:** so theo ngày lịch; hỗ trợ `date`, `datetime`, chuỗi ISO date/datetime; ngày cuối được tính trọn khi `inclusive_end=True` và bị loại khi là mốc kết thúc độc quyền. Chuỗi ngày sai định dạng bị báo lỗi thay vì cho kết quả lọc từ vựng.
- **Lỗi precision:** `ratio_percent` và `percent_change` trả về số thập phân tuần hoàn nhiều chữ số, trái với rulebook yêu cầu phần trăm 2 chữ số. Đã làm tròn bằng `ROUND_HALF_UP`, giữ độ chính xác cao trước khi lượng tử hóa và tăng precision cho giá trị lớn.
- **Lỗi format unanswerable:** rulebook §7 nêu `reason` không bắt buộc; serializer và evaluator trước đó từ chối câu trả lời không có reason. Đã sửa để chấp nhận thiếu reason, đồng thời nếu reason có mặt thì bắt buộc là chuỗi.
- **Lỗi metric số:** evaluator trước đây so khớp thập phân tuyệt đối nên không bỏ qua sai số dấu phẩy động như §9 yêu cầu. Hiện chấp nhận tối đa một ULP cho số không nguyên ở number/percent và các phần tử số lồng trong list/table; hai Decimal khác nhau nhưng làm tròn về cùng float cũng được chấp nhận. Số nguyên (bao gồm tiền/số đếm) vẫn so chính xác. Không xem đây là công thức PA chính thức.
- **Thiếu COUNT DISTINCT:** `COUNT` trước đây chỉ đếm dòng hoặc giá trị không NULL, dễ đếm thừa số đơn/khách khi bảng có nhiều dòng cho một đối tượng. Đã bổ sung `count_distinct` xác định, kiểm tra cột bắt buộc, bỏ NULL và giới hạn cardinality để kiểm soát RAM; prompt adapter hướng planner dùng mã đối tượng khi có nhiều dòng/đối tượng.
- **Lỗi tích hợp phép tính:** QueryPlan có thể tính tỷ lệ từ nhiều aggregation nhưng pipeline yêu cầu scalar result chỉ có một cột, làm câu hỏi như tỷ lệ đơn huỷ bị lỗi dù executor tính đúng. Đã chọn kết quả calculation cuối làm giá trị scalar và thêm integration test tỷ lệ đơn huỷ / mọi đơn duy nhất trên SQLite với đơn nhiều dòng và nhiều trạng thái.
- **Thiếu phân biệt khóa nhóm hiển thị và khóa tie-break:** table trước đây xuất toàn bộ `group_by`, khiến mã sản phẩm thêm để sắp xếp phá hòa lộ vào đáp án. Pipeline/serializer/evaluator hiện lấy `selected_columns` làm cột hiển thị, giữ khóa bổ sung chỉ cho group/sort; prompt planner nêu thứ tự xếp hạng metric desc, mã asc hoặc tên asc khi không có mã. Chỉ kiểm chứng trên fixture/stub, chưa xác minh model thật tự tạo tie-break đúng.
- **Lỗi ghép alias dòng bảng:** evaluator ghép tham lam các dòng bảng không theo thứ tự; alias giao nhau có thể khiến nó bỏ sót một phép ghép hợp lệ và đánh sai đáp án. Đã thay bằng ghép hai phía trên graph tương thích và thêm test hồi quy cho alias chồng lấp.
- **Test hồi quy:** date và datetime đã chuẩn hóa, timestamp cuối ngày, cận kết thúc độc quyền, ISO text và giá trị ngày sai định dạng; kiểm tra phần trăm âm/nửa lên, min/max/top-K tie-break, period scope, time bucket calendar, cận kết thúc độc quyền khi lấp bucket, giới hạn số group, SUM NULL so với bucket trống SUM=0, tolerance one-ULP và cùng-float Decimal cho number/percent/list/table, serialization/evaluator của bảng rỗng và unanswerable không reason, kỳ ngoài độ phủ, kỳ chéo độ phủ và sum 0 trong kỳ có dữ liệu.
- **Kết quả gần nhất:** count-distinct unit/plan/adapter/SQLite **5 passed**; cancellation-rate integration **1 passed**; targeted time-bucket/coverage tests **14 passed**; targeted submission/evaluator tests **24 passed**; tie-break projection/prompt/evaluator **10 passed**; alias-overlap table **3 passed**; kiểm tra float/percent **4 passed**, list/table **1 passed**, Decimal cùng float **1 passed**; full suite `pytest -q` **159 passed**; `compileall -q src tests scripts` và `git diff --check` thành công.

## Việc còn yếu / blocker

1. Chưa có model checkpoint phù hợp, prediction trên tập đánh giá hoặc annotation QueryPlan; vì vậy chưa thể đo accuracy hay khẳng định năng lực question-to-plan.
2. Bucket trống hỗ trợ nhiều aggregation cơ bản nhưng chưa lấp bucket cho chuỗi có thêm chiều group (ví dụ mỗi sản phẩm từng tháng); cần quy tắc domain đầy đủ và test riêng.
3. Cần thực thi từng business rule có dữ liệu và test đối chứng; hiện rulebook chủ yếu được đưa vào prompt nhưng các luật chưa biểu diễn được vẫn bị fail-closed.
4. Cần bộ benchmark/annotation có thể kiểm chứng cho 70 trường hợp unanswerable và câu hỏi mơ hồ/thiếu trường; không suy ra cách giải từ answer gold.
5. Chưa xác minh được evaluator PA/validator challenge-specific và yêu cầu submission; chưa có GPU/model để inference benchmark hoặc fine-tune. Public test chưa được mở và không được dùng.

**Phạm vi các vòng Solve-First đã kiểm chứng:** sửa temporal filter, thêm period scope cho từng aggregation, time bucket theo lịch và bucket rỗng có giới hạn, projection list/table, kiểm tra độ phủ ngày, precision phần trăm, format unanswerable theo rulebook, tolerance số theo rulebook, count-distinct cho khóa lặp và scalar calculation từ nhiều aggregate; tất cả có unit/integration/evaluator tests tương ứng. Không fine-tune, không mở public test hoặc tạo nhãn giả. Đây là năng lực executor/pipeline đã kiểm chứng trên fixture, chưa phải benchmark giải câu hỏi end-to-end.
