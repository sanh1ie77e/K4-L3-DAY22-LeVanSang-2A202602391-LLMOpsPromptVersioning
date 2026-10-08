"""Kiểm tra evidence tại máy; --langsmith xác nhận số trace thật trên server."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import config  # Cấu hình trước LangSmith.

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_FILES = [
    "01_langsmith_traces.png", "02_prompt_hub.png", "02_ab_routing_log.txt",
    "03_ragas_scores.png", "03_ragas_report.json", "04_pii_demo_log.txt",
    "04_json_demo_log.txt",
]
METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def check_local() -> bool:
    """Không dùng file placeholder hoặc JSON thiếu metric làm bằng chứng."""
    ok = True
    for filename in EVIDENCE_FILES:
        path = ROOT / "evidence" / filename
        present = path.is_file() and path.stat().st_size > 0
        print(f"{'✅' if present else '❌'} {filename}")
        ok = ok and present
    tracked_env = subprocess.run(
        ["git", "ls-files", "--", ".env"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"{'❌' if tracked_env else '✅'} .env {'bị track' if tracked_env else 'không bị track'}")
    ok = ok and not bool(tracked_env)
    report_path = ROOT / "evidence" / "03_ragas_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for version in ["prompt_v1_scores", "prompt_v2_scores"]:
                for metric in METRICS:
                    score = report[version][metric]
                    if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError(f"Điểm {version}.{metric} không hợp lệ")
            target = max(report[version]["faithfulness"] for version in
                         ["prompt_v1_scores", "prompt_v2_scores"]) >= 0.8
            if not target or report.get("target_met") is not True:
                raise ValueError("Faithfulness chưa đạt 0.8")
            if report.get("num_questions_per_version") != 50:
                raise ValueError("Báo cáo chưa xác nhận 50 QA cho mỗi phiên bản")
            print("✅ RAGAS: đủ 4 metric, 50 QA mỗi phiên bản và đạt faithfulness ≥ 0.8")
        except (ValueError, KeyError, TypeError) as exc:
            print(f"❌ RAGAS: {exc}")
            ok = False
    return ok


def check_langsmith() -> bool:
    """Đếm trace cha thành công theo đúng hai tên, không đếm run con."""
    if not config.validate():
        return False
    from langsmith import Client
    client = Client(api_key=config.LANGSMITH_API_KEY)
    counts = {"rag-query": 0, "ab-rag-query": 0}
    for run in client.list_runs(project_name=config.LANGSMITH_PROJECT, is_root=True, error=False):
        if run.name in counts:
            counts[run.name] += 1
    for name, count in counts.items():
        print(f"{'✅' if count >= 50 else '❌'} {name}: {count} traces thành công")
    project = client.read_project(project_name=config.LANGSMITH_PROJECT)
    verification = {
        "project": config.LANGSMITH_PROJECT, "project_id": str(project.id),
        "counts": counts, "target_met": all(count >= 50 for count in counts.values()),
    }
    project_url = getattr(project, "url", None)
    if project_url:
        verification["project_url"] = project_url
        print(f"LangSmith project: {project_url}")
    (ROOT / "evidence" / "langsmith_verification.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return verification["target_met"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langsmith", action="store_true", help="Kiểm tra thêm traces trên server")
    args = parser.parse_args()
    ok = check_local()
    if args.langsmith:
        ok = check_langsmith() and ok
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
