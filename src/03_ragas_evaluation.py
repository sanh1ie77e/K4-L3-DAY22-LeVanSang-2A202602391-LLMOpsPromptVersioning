"""Bước 3: đánh giá đủ 50 QA cho mỗi prompt bằng bốn metric RAGAS."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config  # Phải import trước LangChain.

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from qa_pairs import QA_PAIRS
from utils.data_loader import build_vectorstore, load_knowledge_base, split_text
from utils.evidence import tee_logs
from utils.llm_factory import get_embeddings, get_llm


# Giống hệt SYSTEM_V1 / SYSTEM_V2 của bước 2, kể cả {context}.
SYSTEM_V1 = (
    "You are a helpful AI study assistant. Answer the question directly in 2–4 "
    "short sentences, using only facts supported by the supplied context. "
    "Use the same language as the question and avoid unrelated details. "
    "If the context does not contain the answer, say that the available context "
    "is insufficient instead of guessing.\n\nContext:\n{context}"
)
SYSTEM_V2 = (
    "You are an AI subject expert teaching a student. Identify the facts in the "
    "context that answer the question and organize your response as 'Core idea:' "
    "followed by 'Explanation:'. Give a precise definition first, then explain "
    "the relevant mechanism, distinction, or limitation in 3–5 concise sentences. "
    "Use only information explicitly supported by the context, preserve "
    "technical terms, and answer in the language of the question. "
    "If evidence is missing, state what cannot be determined; do not invent "
    "examples, numbers, or facts.\n\nContext:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1), ("human", "{question}"),
])
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2), ("human", "{question}"),
])
PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def setup_vectorstore():
    """Giữ cùng FAISS/chunking/retriever với hai bước trước."""
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    return build_vectorstore(chunks, embeddings)


def run_rag(retriever, llm, prompt, question: str) -> dict:
    """contexts là list[str]; ctx_str chỉ dùng làm input cho prompt."""
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": ctx_str, "question": question,
    })
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """Chạy toàn bộ QA_PAIRS, giữ nguyên reference để chấm context_*."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    prompt = PROMPTS[prompt_version]
    results = []
    print(f"\n🚀 Đang chạy {len(QA_PAIRS)} câu hỏi với prompt {prompt_version} ...")
    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question": qa["question"],
            "reference": qa["reference"],
            "answer": out["answer"],
            "contexts": out["contexts"],
        })
        print(f"  [{i:02d}/{len(QA_PAIRS)}] {qa['question']}")
    return results


def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """Ánh xạ đúng bốn trường SingleTurnSample, không ghép retrieved_contexts."""
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


def run_ragas_eval(rag_results: list, version: str) -> dict:
    """Chấm bốn metric; từ chối báo cáo thiếu điểm hoặc chứa NaN."""
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version.upper()} ...")
    dataset = build_ragas_dataset(rag_results)
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=RunConfig(max_workers=4, timeout=180, max_retries=3),
        raise_exceptions=True,
    )
    scores = {}
    for key in METRIC_NAMES:
        raw = result[key]
        if len(raw) != len(rag_results) or any(
            value is None or not np.isfinite(value) for value in raw
        ):
            raise ValueError(f"Metric {key} của {version} thiếu điểm hợp lệ; cần chạy lại.")
        scores[key] = float(np.mean(raw))
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for key, score in scores.items():
        star = " ⭐" if key == "faithfulness" and score >= 0.8 else ""
        print(f"  {key:30s}: {score:.4f}{star}")
    return scores


def write_analysis(report: dict, evidence_path: Path):
    """Viết phân tích từ điểm thực tế; tách nhận xét số liệu và giả thuyết."""
    v1, v2 = report["prompt_v1_scores"], report["prompt_v2_scores"]
    rows = []
    for metric in METRIC_NAMES:
        winner = "V1" if v1[metric] > v2[metric] else "V2" if v2[metric] > v1[metric] else "Tie"
        rows.append(f"| {metric} | {v1[metric]:.4f} | {v2[metric]:.4f} | {winner} |")
    faith_winner = (
        "V1" if v1["faithfulness"] > v2["faithfulness"] else
        "V2" if v2["faithfulness"] > v1["faithfulness"] else "Hai phiên bản bằng nhau"
    )
    text = (
        "# Evidence — LeVanSang-2A202602391\n\n"
        f"Thời điểm đánh giá: {report['evaluated_at']} (GMT+7).\n\n"
        "## Thiết kế thí nghiệm\n\n"
        "V1 trả lời trực tiếp trong 2–4 câu. V2 nêu Core idea rồi Explanation "
        "trong 3–5 câu. Cả hai chỉ dùng context và thừa nhận khi thiếu dữ liệu. "
        "System prompt của bước 2 và bước 3 giống hệt nhau.\n\n"
        "Knowledge base, FAISS, chunk size 500, overlap 50 và k=3 được giữ cố định. "
        "A/B routing dùng MD5: V1=19, V2=31. Đánh giá RAGAS dùng đủ 50 QA "
        "cho mỗi phiên bản.\n\n"
        "## Kết quả đo\n\n"
        "| Metric | V1 | V2 | Cao hơn |\n|---|---:|---:|---|\n"
        + "\n".join(rows) + "\n\n"
        + f"Faithfulness cao nhất: {max(v1['faithfulness'], v2['faithfulness']):.4f}; "
        + f"đạt ngưỡng 0.8: {report['target_met']}. Kết quả so sánh faithfulness: {faith_winner}.\n\n"
        + "## Phân tích\n\n"
        + "Hai phiên bản nhận cùng kết quả retrieval cho từng câu hỏi. Khác biệt "
        + "faithfulness/relevancy phản ánh nội dung câu trả lời và biến động của "
        + "LLM chấm điểm. V1 giới hạn độ dài nên có thể giảm số claim không được "
        + "context hỗ trợ; V2 giải thích có cấu trúc có thể giúp câu trả lời đủ ý "
        + "hơn, nhưng mỗi chi tiết thêm cũng cần bằng chứng. Đây là giả thuyết "
        + "giải thích, cần đối chiếu trace/câu trả lời để xác nhận nguyên nhân.\n\n"
        + "Context recall/precision tập trung vào retrieval so với reference; "
        + "chênh lệch giữa hai lần chấm không chứng minh prompt đã làm retriever "
        + "tốt hơn. Một lần chạy cũng chưa đủ để kết luận chênh lệch nhỏ có ý nghĩa "
        + "thống kê.\n\n"
        + "## Bằng chứng\n\n"
        + "Báo cáo số liệu: 03_ragas_report.json; log bảng điểm: "
        + "03_ragas_evaluation_log.txt; log routing: 02_ab_routing_log.txt. "
        + "Hai log Guardrails được tạo từ 6 case PII và 5 case JSON. "
        + "Ba ảnh bắt buộc phải chụp từ LangSmith/terminal thực tế; kiểm tra "
        + "đủ ảnh trước khi nộp. Unit test không thay thế traces hoặc điểm RAGAS.\n"
    )
    (evidence_path.parent / "README.md").write_text(text, encoding="utf-8")


def _run():
    print("=" * 65)
    print("  Bước 3: RAGAS Evaluation — LeVanSang-2A202602391")
    print("=" * 65)
    if not config.validate():
        sys.exit(1)
    vectorstore = setup_vectorstore()
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in METRIC_NAMES:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "V1" if s1 > s2 else "V2" if s2 > s1 else "Tie"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️ Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Kiểm tra context và facts trong câu trả lời trước khi chỉnh retrieval.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "num_questions_per_version": len(QA_PAIRS),
        "student": "LeVanSang-2A202602391",
        "langsmith_project": config.LANGSMITH_PROJECT,
        "evaluated_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
        "system_v1": SYSTEM_V1,
        "system_v2": SYSTEM_V2,
    }
    report_path = Path(__file__).resolve().parent.parent / "data" / "ragas_report.json"
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    report_path.write_text(payload, encoding="utf-8")
    evidence_path = report_path.parent.parent / "evidence" / "03_ragas_report.json"
    evidence_path.write_text(payload, encoding="utf-8")
    write_analysis(report, evidence_path)
    print(f"💾 Đã lưu báo cáo vào {report_path} và {evidence_path}")


def main():
    with tee_logs("03_ragas_evaluation_log.txt"):
        _run()


if __name__ == "__main__":
    main()
