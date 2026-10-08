"""In lại bảng điểm đã đo để chụp evidence, không gọi API hoặc chấm lại."""
import json
import math
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    report_path = root / "evidence" / "03_ragas_report.json"
    if not report_path.is_file():
        raise SystemExit("Chưa có báo cáo RAGAS. Hãy chạy bước 3 trước.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    v1, v2 = report["prompt_v1_scores"], report["prompt_v2_scores"]
    if report["num_questions_per_version"] != 50 or any(
        not math.isfinite(scores[metric]) or not 0 <= scores[metric] <= 1
        for scores in (v1, v2) for metric in metrics
    ):
        raise SystemExit("Báo cáo phải đủ 50 QA mỗi phiên bản và điểm hợp lệ.")
    print(f"RAGAS | {report['student']} | 50 QA / prompt")
    print(f"Thời điểm đo: {report['evaluated_at']}")
    print(f"Nguồn: {report_path}")
    print("=" * 65)
    print(f"{'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in metrics:
        a, b = v1[metric], v2[metric]
        winner = "V1" if a > b else "V2" if b > a else "Tie"
        print(f"{metric:30s}  {a:8.4f}  {b:8.4f}  {winner}")
    best = max(v1["faithfulness"], v2["faithfulness"])
    print(f"\n{'✅ Đạt mục tiêu' if best >= 0.8 else '⚠️ Chưa đạt mục tiêu'}: faithfulness = {best:.4f}")


if __name__ == "__main__":
    main()
