# Evidence — LeVanSang-2A202602391

Thời điểm đánh giá: 2026-10-08T12:49:03.418123+07:00 (GMT+7).

## Thiết kế thí nghiệm

V1 trả lời trực tiếp trong 2–4 câu. V2 nêu Core idea rồi Explanation trong 3–5 câu. Cả hai chỉ dùng context và thừa nhận khi thiếu dữ liệu. System prompt của bước 2 và bước 3 giống hệt nhau.

Knowledge base, FAISS, chunk size 500, overlap 50 và k=3 được giữ cố định. A/B routing dùng MD5: V1=19, V2=31. Đánh giá RAGAS dùng đủ 50 QA cho mỗi phiên bản.

## Kết quả đo

| Metric | V1 | V2 | Cao hơn |
|---|---:|---:|---|
| faithfulness | 0.9781 | 0.9059 | V1 |
| answer_relevancy | 0.9224 | 0.8534 | V1 |
| context_recall | 1.0000 | 1.0000 | Tie |
| context_precision | 0.9450 | 0.9417 | V1 |

Faithfulness cao nhất: 0.9781; đạt ngưỡng 0.8: True. Kết quả so sánh faithfulness: V1.

## Phân tích

Hai phiên bản nhận cùng kết quả retrieval cho từng câu hỏi. Khác biệt faithfulness/relevancy phản ánh nội dung câu trả lời và biến động của LLM chấm điểm. V1 giới hạn độ dài nên có thể giảm số claim không được context hỗ trợ; V2 giải thích có cấu trúc có thể giúp câu trả lời đủ ý hơn, nhưng mỗi chi tiết thêm cũng cần bằng chứng. Đây là giả thuyết giải thích, cần đối chiếu trace/câu trả lời để xác nhận nguyên nhân.

Context recall/precision tập trung vào retrieval so với reference; chênh lệch giữa hai lần chấm không chứng minh prompt đã làm retriever tốt hơn. Một lần chạy cũng chưa đủ để kết luận chênh lệch nhỏ có ý nghĩa thống kê.

## Bằng chứng

Báo cáo số liệu: 03_ragas_report.json; log bảng điểm: 03_ragas_evaluation_log.txt; log routing: 02_ab_routing_log.txt. Hai log Guardrails được tạo từ 6 case PII và 5 case JSON. Ba ảnh bắt buộc phải chụp từ LangSmith/terminal thực tế; kiểm tra đủ ảnh trước khi nộp. Unit test không thay thế traces hoặc điểm RAGAS.
