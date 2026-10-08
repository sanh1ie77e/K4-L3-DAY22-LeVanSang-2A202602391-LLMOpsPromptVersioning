"""Bước 4: tự viết PII/JSON validators; demo độc lập, không gọi LLM."""
import ast
import json
import os
import re

os.environ.setdefault("GUARDRAILS_RUN_SYNC", "true")

from guardrails import Guard, settings
from guardrails.validators import FailResult, PassResult, Validator, register_validator
from guardrails.validator_base import OnFailAction

from utils.evidence import tee_logs

# Bước này chỉ cần kiểm tra chuỗi tại máy, không cần telemetry.
settings.disable_tracing = True
settings.rc.enable_metrics = False


@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """Dò bốn loại PII bằng regex và trả fix_value để Guard thay đầu ra."""

    PII_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "PHONE": r"(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        redacted_text = value
        found_pii = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)
            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))
        if found_pii:
            print(f"  ⚠️ Đã redact {len(found_pii)} PII: {[p[0] for p in found_pii]}")
            return FailResult(error_message="Phát hiện PII", fix_value=redacted_text)
        return PassResult()


@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """Sửa fences/nháy đơn/trailing comma; luôn có JSON dự phòng."""

    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()
        # Chỉ đổi nháy đơn bao quanh chuỗi, giữ apostrophe trong chuỗi nháy đôi.
        string_pattern = r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*' '''.strip()

        def normalize_quotes(match):
            token = match.group(0)
            if token.startswith('"'):
                return token
            return json.dumps(ast.literal_eval(token), ensure_ascii=False)

        text = re.sub(string_pattern, normalize_quotes, text)
        # Bỏ trailing comma ở ngoài chuỗi, không sửa dấu phẩy thuộc dữ liệu.
        text = re.sub(
            r'("(?:\\.|[^"\\])*")|,\s*([}\]])',
            lambda match: match.group(1) or match.group(2),
            text,
        )
        return text

    def validate(self, value: str, metadata: dict):
        try:
            json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            pass
        try:
            parsed = json.loads(self._repair(value))
            print("  🔧 JSON đã được sửa thành công")
            return FailResult(
                error_message="JSON lỗi, đã tự sửa",
                fix_value=json.dumps(parsed, indent=2, ensure_ascii=False),
            )
        except (json.JSONDecodeError, SyntaxError, ValueError):
            fallback = json.dumps(
                {"error": "Không thể phân tích JSON", "raw": value[:200]},
                ensure_ascii=False,
            )
            return FailResult(error_message="Không thể sửa JSON", fix_value=fallback)


def demo_pii_guard():
    print("\n" + "=" * 55)
    print("  Demo: PII Detection & Redaction")
    print("=" * 55)
    guard = Guard()
    guard.configure(allow_metrics_collection=False)
    settings.rc.enable_metrics = False
    guard.use(PIIDetector(on_fail=OnFailAction.FIX))
    test_cases = [
        ("Email", "Contact John at john.doe@example.com for details."),
        ("Phone", "Call our support line at (555) 867-5309."),
        ("SSN", "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII", "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean", "No sensitive information in this text."),
    ]
    for label, text in test_cases:
        result = guard.validate(text)
        output = result.validated_output
        if label == "Clean":
            assert output == text, "Chuỗi sạch phải giữ nguyên"
        else:
            assert output is not None and output != text, "PII chưa bị che"
            assert not any(re.search(pattern, output) for pattern in PIIDetector.PII_PATTERNS.values())
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {output}")


def demo_json_guard():
    print("\n" + "=" * 55)
    print("  Demo: JSON Formatting & Repair")
    print("=" * 55)
    guard = Guard()
    guard.configure(allow_metrics_collection=False)
    settings.rc.enable_metrics = False
    guard.use(JSONFormatter(on_fail=OnFailAction.FIX))
    test_cases = [
        ("Valid JSON", '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```'),
        ("Single quotes", "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma", '{"key": "value",}'),
        ("Truly invalid", "This is not JSON at all: ??? {]"),
    ]
    for label, text in test_cases:
        result = guard.validate(text)
        output = result.validated_output
        parsed = json.loads(output)
        if label == "Truly invalid":
            assert "error" in parsed and "raw" in parsed, "Thiếu JSON dự phòng"
        print(f"\n[{label}] ✅ JSON hợp lệ")
        print(f"  Input:  {text}")
        print(f"  Output: {output}")


def _run():
    print("=" * 55)
    print("  Bước 4: Guardrails AI — LeVanSang-2A202602391")
    print("=" * 55)
    demo_pii_guard()
    demo_json_guard()
    print("\n✅ Bước 4 hoàn thành: 6 PII cases và 5 JSON cases.")


def main():
    # Hai file cùng lưu cả hai demo như lệnh tee trong slide.
    with tee_logs("04_pii_demo_log.txt", "04_json_demo_log.txt"):
        _run()


if __name__ == "__main__":
    main()
