# Evidence — LeVanSang-2A202602391

## Thiết kế thí nghiệm

V1 trả lời trực tiếp trong 2–4 câu. V2 nêu `Core idea:` rồi `Explanation:`
trong 3–5 câu, tập trung vào định nghĩa, cơ chế hoặc giới hạn được context hỗ trợ.
Cả hai chỉ dùng context, trả lời bằng ngôn ngữ của câu hỏi và thừa nhận khi
thiếu dữ liệu. Bước 2 và 3 sử dụng hai system prompt giống hệt nhau.

Knowledge base, chunk size 500, overlap 50 và retriever k=3 được giữ cố định.
Bước 2 chia 50 request bằng MD5 (V1=19, V2=31). Bước 3 đánh giá toàn bộ
50 QA qua từng phiên bản, nên mỗi phiên bản có 50 mẫu để so sánh.

## Trạng thái bằng chứng

Hai log `04_pii_demo_log.txt` và `04_json_demo_log.txt` được tạo từ lần chạy
Guardrails thực tế: 6 case PII và 5 case JSON. Bốn loại PII đều bị che,
chuỗi sạch được giữ nguyên, JSON sửa được và JSON không sửa được đều có
đầu ra parse được.

Chưa có kết quả RAGAS hoặc ảnh LangSmith khi chưa chạy với API key hợp lệ.
Không suy ra điểm faithfulness từ unit test và không dùng số minh họa của slide
làm kết quả bài nộp. Phân tích định lượng sẽ được ghi ở đây khi bước 3 hoàn thành.

## Bằng chứng bắt buộc

| File | Cách tạo |
|---|---|
| `01_langsmith_traces.png` | Chụp LangSmith với ≥50 traces `rag-query`, kiểm tra 1 trace có 3 documents |
| `02_prompt_hub.png` | Chụp hai prompt cá nhân đã push/pull trên Prompt Hub |
| `02_ab_routing_log.txt` | Tự tạo khi bước 2 chạy thành công với Hub |
| `03_ragas_scores.png` | Chụp bảng so sánh V1/V2 từ terminal sau bước 3 |
| `03_ragas_report.json` | Bước 3 tự sao chép báo cáo thật từ `data/` |
| `04_pii_demo_log.txt` | Tự tạo bởi bước 4 |
| `04_json_demo_log.txt` | Tự tạo bởi bước 4 |

Không đưa API key vào code, log hoặc ảnh chụp màn hình.
