# Bài làm — LeVanSang-2A202602391

- Tên repo khi nộp: `K4-L3-DAY22-LeVanSang-2A202602391-LLMOpsPromptVersioning`.
- Project LangSmith trong `.env`: [day22-LeVanSang-2A202602391](https://smith.langchain.com/o/a45bf19d-9bec-4923-a77b-4d2b4cdfb553/projects/p/4078a719-000c-4abf-a07b-f1a6744850a4).
- Prompt Hub: `le-van-sang-2a202602391-rag-prompt-v1` và `le-van-sang-2a202602391-rag-prompt-v2`.
- V1 trả lời ngắn gọn; V2 định nghĩa và giải thích có cấu trúc. Hai prompt được giữ giống hệt giữa bước 2 và bước 3.

## Chạy bài làm trên Windows PowerShell

Từ thư mục gốc repo:

```powershell
python -m venv venv                 # chỉ khi chưa có venv
.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python -m pip install -r requirements.txt
# Chỉ chép nếu chưa có .env; giữ nguyên key nếu file đã tồn tại.
if (!(Test-Path .env)) { Copy-Item .env.example .env }
# Điền LANGCHAIN_API_KEY và OPENAI_API_KEY trong .env, không dán vào code.
python src/config.py
python -m unittest discover -s tests -v
python src/run_all.py
python src/check_submission.py --langsmith  # sau khi bổ sung đủ 3 ảnh
```

Chạy riêng một bước: `python src/run_all.py --step 4` (thay số bằng 1–4).
Với Git Bash, dùng `source venv/Scripts/activate` và `export PYTHONUTF8=1`.

Các bước 2 và 4 tự lưu log UTF-8 vào đúng tên file evidence. Bước 3 tự lưu cả
`data/ragas_report.json` và bản sao `evidence/03_ragas_report.json`; chỉ ghi báo cáo
khi cả hai phiên bản có đủ bốn metric và không có điểm thiếu/NaN.
Log đầy đủ của bước 3 được lưu ở `evidence/03_ragas_evaluation_log.txt`.
Nếu đã đóng terminal, chạy `python src/show_ragas_report.py` để in lại bảng
điểm từ báo cáo đã đo, không gọi API hay đánh giá lại. Chụp bảng này và lưu
thành `evidence/03_ragas_scores.png`.
Chương trình bước 2 bắt buộc pull thành công từ Hub trong lần chạy nộp bài;
nhánh fallback local chỉ phục vụ debug khi gọi hàm riêng.

Sau khi chạy thật, chụp ba ảnh theo bảng evidence bên dưới. Số traces phải
được xác nhận trên LangSmith; dòng hoàn thành trên terminal không thay thế
bằng chứng đó. Test tại máy dùng model giả để kiểm tra logic và không được
tính là traces hay điểm RAGAS của bài nộp.

Guardrails dùng chế độ đồng bộ và tắt telemetry cho demo tại máy. Validator
vẫn chạy qua `Guard.validate()` với `OnFailAction.FIX` trong constructor.
Regex PII minh họa email, số điện thoại kiểu Mỹ, SSN và thẻ 16 chữ số;
không phải bộ phát hiện đầy đủ cho mọi định dạng PII.

`requirements-lock.txt` lưu phiên bản của môi trường Windows/Python 3.13 đã
kiểm tra. Khi cần tái tạo đúng môi trường này, cài bằng
`python -m pip install -r requirements-lock.txt`.

---

> **📌 Hình thức: BÀI CÁ NHÂN** — mỗi học viên tự làm và tự nộp 1 repo theo quy ước đặt tên.
> **⏰ Thời lượng:** ~3–4 giờ · **Deadline:** 23:59 ngày học lab (GMT+7)
>
> | Tài liệu | Nội dung |
> |---|---|
> | [CHECKPOINTS.md](CHECKPOINTS.md) | **Hướng dẫn làm bài từng bước**: cần làm gì, sản phẩm, cách tự kiểm tra |
> | [RUBRIC.md](RUBRIC.md) | Tiêu chí chấm điểm, điểm thưởng (tối đa +10) |
> | [SUBMISSION.md](SUBMISSION.md) | Tên repo, cấu trúc nộp bài, nơi nộp, deadline |
> | [RULES.md](RULES.md) | Quy định sử dụng AI, sao chép, nộp muộn, bảo mật API key |

# Chào mừng các bạn đến với Day 22: LangSmith + Prompt Versioning

## Tổng quan

Trong lab này, bạn sẽ xây dựng một hệ thống hỏi đáp hoàn chỉnh tích hợp nhiều công nghệ AI hiện đại:

- **RAG Pipeline**: Xây dựng pipeline Retrieval-Augmented Generation sử dụng FAISS làm vector store và LangChain để kết nối các thành phần.
- **LangSmith Tracing**: Theo dõi và quan sát toàn bộ luồng xử lý của ứng dụng LLM thông qua LangSmith dashboard.
- **Prompt Hub & A/B Testing**: Quản lý phiên bản prompt trên LangSmith Prompt Hub và thực hiện A/B routing để so sánh hiệu quả giữa các phiên bản.
- **RAGAS Evaluation**: Đánh giá chất lượng hệ thống RAG theo 4 chỉ số định lượng: faithfulness, answer relevancy, context recall, context precision.
- **Guardrails AI**: Triển khai các bộ kiểm duyệt tự động để phát hiện thông tin cá nhân (PII) và sửa lỗi định dạng JSON trong đầu ra của LLM.

---

## Mục tiêu học tập

Sau khi hoàn thành lab này, bạn sẽ có thể:

- Xây dựng và triển khai RAG pipeline hoàn chỉnh với LangChain LCEL và FAISS vector store.
- Tích hợp LangSmith để theo dõi, gỡ lỗi và phân tích hiệu suất của ứng dụng LLM trong thực tế.
- Quản lý vòng đời prompt bằng LangSmith Prompt Hub và thực hiện A/B testing có kiểm soát.
- Đánh giá hệ thống RAG một cách định lượng bằng framework RAGAS với các chỉ số chuẩn công nghiệp.
- Áp dụng Guardrails AI để xây dựng validator tùy chỉnh nhằm bảo vệ đầu ra của LLM khỏi dữ liệu nhạy cảm và lỗi định dạng.

---

## Yêu cầu trước

Trước khi bắt đầu, hãy đảm bảo bạn đã có:

- **Python 3.10 trở lên** — kiểm tra bằng lệnh `python --version`
- **API key** của ít nhất một trong các nhà cung cấp LLM sau:
  - OpenAI (`OPENAI_API_KEY`)
  - Google Gemini (`GOOGLE_API_KEY`)
  - Anthropic Claude (`ANTHROPIC_API_KEY`)
  - OpenRouter (`OPENROUTER_API_KEY`)
  - Ollama (chạy local, không cần API key)
- **Tài khoản LangSmith** — đăng ký miễn phí tại [smith.langchain.com](https://smith.langchain.com) và lấy API key

---

## Cài đặt nhanh

```bash
pip install -r requirements.txt
pip install "langchain-community<0.4"   # bắt buộc: bản 0.4 làm import ragas lỗi
cp .env.example .env             # điền LANGCHAIN_API_KEY, PROVIDER và key của provider
cd src && python config.py       # phải in: ✅ Config OK
```

Hướng dẫn chi tiết (tạo venv, lấy API key LangSmith, chọn provider, lưu ý cho Windows) ở **Checkpoint 0** trong [CHECKPOINTS.md](CHECKPOINTS.md).

---

## Cấu trúc dự án

```
Lab/
├── src/
│   ├── config.py                      # Tải .env, cấu hình providers
│   ├── utils/
│   │   ├── llm_factory.py             # Factory tạo LLM và Embeddings (5 providers)
│   │   └── data_loader.py             # Load knowledge base, chunk, build FAISS
│   ├── qa_pairs.py                    # 50 cặp câu hỏi + đáp án chuẩn
│   ├── 01_langsmith_rag_pipeline.py   # Bước 1: RAG + LangSmith tracing
│   ├── 02_prompt_hub_ab_routing.py    # Bước 2: Prompt Hub + A/B routing
│   ├── 03_ragas_evaluation.py         # Bước 3: RAGAS evaluation (~15-30 phút)
│   ├── 04_guardrails_validator.py     # Bước 4: Guardrails AI validators
│   └── run_all.py                     # Chạy tất cả các bước
├── data/
│   ├── knowledge_base.txt             # Tài liệu nguồn cho RAG
│   └── ragas_report.json              # Được tạo ra ở Bước 3
├── evidence/                          # Nộp thư mục này lên GitHub
│   ├── 01_langsmith_traces.png
│   ├── 02_prompt_hub.png
│   ├── 02_ab_routing_log.txt
│   ├── 03_ragas_scores.png
│   ├── 03_ragas_report.json
│   ├── 04_pii_demo_log.txt
│   └── 04_json_demo_log.txt
├── .env.example                        # Template biến môi trường
├── requirements.txt
├── README.md                       # Tổng quan (file này)
├── CHECKPOINTS.md                  # Hướng dẫn làm bài từng bước
├── RUBRIC.md                       # Tiêu chí chấm điểm
├── SUBMISSION.md                   # Cách nộp bài
└── RULES.md                        # Quy định làm bài
```

---

## Các nhiệm vụ

Lab được chia thành 4 nhiệm vụ, mỗi nhiệm vụ 25 điểm (tổng 100 điểm):

| Nhiệm vụ | Tên                              | Điểm | Thời gian ước tính   |
|----------|----------------------------------|------|----------------------|
| 1        | RAG Pipeline với LangSmith       | 25đ  | 25–45 phút           |
| 2        | Prompt Hub & A/B Routing         | 25đ  | 20–30 phút           |
| 3        | RAGAS Evaluation                 | 25đ  | 45–75 phút           |
| 4        | Guardrails AI Validators         | 25đ  | 20–30 phút           |

**Nhiệm vụ 1 — RAG Pipeline với LangSmith (25đ):** Xây dựng vector store từ knowledge base, tạo RAG chain, và tích hợp `@traceable` để ghi lại ít nhất 50 traces trên LangSmith dashboard.

**Nhiệm vụ 2 — Prompt Hub & A/B Routing (25đ):** Soạn 2 system prompt có ngữ nghĩa khác biệt, đẩy lên LangSmith Prompt Hub, pull về khi chạy, và định tuyến câu hỏi theo hash của `request_id`.

**Nhiệm vụ 3 — RAGAS Evaluation (25đ):** Chạy 50 cặp QA qua cả 2 phiên bản prompt, xây dựng `EvaluationDataset`, tính 4 chỉ số RAGAS, và đạt faithfulness ≥ 0.8 với ít nhất 1 phiên bản.

**Nhiệm vụ 4 — Guardrails AI Validators (25đ):** Triển khai `PIIDetector` tự động che thông tin cá nhân và `JSONFormatter` tự động sửa JSON lỗi từ đầu ra của LLM.

---

Cách làm từng nhiệm vụ: xem [CHECKPOINTS.md](CHECKPOINTS.md). Cách nộp bài: xem [SUBMISSION.md](SUBMISSION.md).

---

## Tips và lưu ý

**LangSmith tracing — đặt biến môi trường đúng thứ tự:**
Các biến `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, và `LANGCHAIN_PROJECT` phải được đặt **trước khi import bất kỳ thứ gì từ LangChain**. Nếu import trước khi đặt biến, tracing sẽ không hoạt động.

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"   # Phải đặt trước
os.environ["LANGCHAIN_API_KEY"]    = "..."     # Phải đặt trước
from langchain_core.prompts import ChatPromptTemplate  # Sau đó mới import
```

**RAGAS chậm — bắt đầu sớm:**
Bước 3 sẽ mất từ 15 đến 30 phút để hoàn thành do phải gọi LLM cho mỗi sample trong bộ đánh giá. Hãy bắt đầu bước này ngay khi bước 2 xong, đặc biệt nếu bạn đang dùng model có rate limit thấp.

**Guardrails AI — `on_fail` phải truyền đúng chỗ:**
Tham số `on_fail` phải được truyền vào **constructor của validator**, không phải vào `Guard.use()`:

```python
# ĐÚNG
Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

# SAI — sẽ không hoạt động đúng
Guard().use(PIIDetector(), on_fail=OnFailAction.FIX)
```

**Lưu ý phiên bản thư viện:**
- `langchain-community` phải `< 0.4` (chạy `pip install "langchain-community<0.4"` sau khi cài `requirements.txt`): bản 0.4 làm `import ragas` lỗi `No module named 'langchain_community.chat_models.vertexai'`.
- RAGAS 0.4: `result[metric_name]` trả về **list** điểm theo từng sample → dùng `numpy.mean()`; truyền `llm=` và `embeddings=` vào `evaluate()`. Cảnh báo deprecated khi import `ragas.metrics` có thể bỏ qua.
- Guardrails 0.11: với `OnFailAction.FIX`, chỉ `FailResult(fix_value=...)` mới thay được output; `PassResult(value_override=...)` **không** có tác dụng.

**Bảo mật — không bao giờ commit `.env`:**
Tệp `.env` chứa API key nhạy cảm. Đảm bảo `.gitignore` đã có dòng `.env` trước khi push lên GitHub. Chỉ commit tệp `.env.example` (không chứa giá trị thật). Vi phạm quy tắc này sẽ bị trừ 10 điểm tự động.

---

## Tài liệu tham khảo

| Tài liệu                    | Đường dẫn                                                          |
|-----------------------------|--------------------------------------------------------------------|
| LangSmith Docs              | https://docs.smith.langchain.com                                   |
| LangChain LCEL              | https://python.langchain.com/docs/concepts/lcel                    |
| LangSmith Prompt Hub        | https://docs.smith.langchain.com/prompt-hub                        |
| RAGAS Documentation         | https://docs.ragas.io                                              |
| Guardrails AI               | https://www.guardrailsai.com/docs                                  |
| FAISS (Facebook AI)         | https://faiss.ai                                                   |
| LangChain FAISS Integration | https://python.langchain.com/docs/integrations/vectorstores/faiss  |
