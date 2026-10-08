"""Kiểm tra chức năng tại máy; không gọi API và không tạo evidence thí nghiệm."""
import importlib
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

step1 = importlib.import_module("01_langsmith_rag_pipeline")
step2 = importlib.import_module("02_prompt_hub_ab_routing")
step3 = importlib.import_module("03_ragas_evaluation")
step4 = importlib.import_module("04_guardrails_validator")


class RetrieverCapture(BaseCallbackHandler):
    def __init__(self):
        self.documents = []

    def on_retriever_end(self, documents, **kwargs):
        self.documents = documents


class LabTests(unittest.TestCase):
    def test_rag_chain_retrieves_three_docs_and_supplies_context(self):
        texts = [f"Lab fact {i}: FAISS retrieves document {i}." for i in range(4)]
        store = FAISS.from_texts(texts, DeterministicFakeEmbedding(size=24))
        captured = []

        def respond(prompt_value):
            captured.extend(prompt_value.to_messages())
            return AIMessage(content="Grounded answer")

        with patch.object(step1, "get_llm", return_value=RunnableLambda(respond)):
            chain, retriever = step1.build_rag_chain(store)
        callback = RetrieverCapture()
        question = "What does FAISS retrieve?"
        answer = chain.invoke(question, config={"callbacks": [callback]})
        self.assertEqual(answer, "Grounded answer")
        self.assertEqual(len(callback.documents), 3)
        self.assertEqual(retriever.search_kwargs["k"], 3)
        for document in callback.documents:
            self.assertIn(document.page_content, captured[0].content)
        self.assertEqual(captured[1].content, question)

    def test_prompt_versions_are_identical_between_steps(self):
        self.assertEqual(step2.SYSTEM_V1, step3.SYSTEM_V1)
        self.assertEqual(step2.SYSTEM_V2, step3.SYSTEM_V2)
        self.assertNotEqual(step2.SYSTEM_V1, step2.SYSTEM_V2)
        for prompt in [step2.PROMPT_V1, step2.PROMPT_V2]:
            self.assertEqual(set(prompt.input_variables), {"context", "question"})

    def test_routing_is_repeatable_and_both_versions_receive_requests(self):
        first = [step2.get_prompt_version(f"req-{i:04d}") for i in range(50)]
        second = [step2.get_prompt_version(f"req-{i:04d}") for i in range(50)]
        self.assertEqual(first, second)
        self.assertEqual(first.count(step2.PROMPT_V1_NAME), 19)
        self.assertEqual(first.count(step2.PROMPT_V2_NAME), 31)

    def test_hub_pull_uses_remote_templates_and_can_require_hub(self):
        remote = step2.PROMPT_V1.partial()
        client = Mock()
        client.pull_prompt.return_value = remote
        with redirect_stdout(StringIO()):
            prompts = step2.pull_prompts_from_hub(client, allow_fallback=False)
        self.assertIs(prompts[step2.PROMPT_V1_NAME], remote)
        self.assertIs(prompts[step2.PROMPT_V2_NAME], remote)
        client.pull_prompt.side_effect = ConnectionError("Hub unavailable")
        with self.assertRaises(ConnectionError):
            step2.pull_prompts_from_hub(client, allow_fallback=False)

    def test_ragas_preserves_each_context_and_maps_reference(self):
        retriever = RunnableLambda(lambda question: [
            Mock(page_content="First passage"), Mock(page_content="Second passage"),
        ])
        llm = RunnableLambda(lambda prompt: AIMessage(content="Answer"))
        out = step3.run_rag(retriever, llm, step3.PROMPT_V1, "Question")
        self.assertEqual(out["contexts"], ["First passage", "Second passage"])
        dataset = step3.build_ragas_dataset([{
            "question": "Question", "answer": out["answer"],
            "contexts": out["contexts"], "reference": "Reference",
        }])
        sample = dataset.samples[0]
        self.assertEqual(sample.user_input, "Question")
        self.assertEqual(sample.response, "Answer")
        self.assertEqual(sample.retrieved_contexts, out["contexts"])
        self.assertEqual(sample.reference, "Reference")
        with self.assertRaises(ValidationError):
            step3.build_ragas_dataset([{
                "question": "Q", "answer": "A", "contexts": "not a list", "reference": "R",
            }])

    def test_ragas_rejects_nan_instead_of_reporting_partial_mean(self):
        scores = {name: [float("nan")] for name in step3.METRIC_NAMES}
        with patch.object(step3, "evaluate", return_value=scores), \
             patch.object(step3, "get_llm"), patch.object(step3, "get_embeddings"), \
             redirect_stdout(StringIO()):
            with self.assertRaises(ValueError):
                step3.run_ragas_eval([{
                    "question": "Q", "answer": "A", "contexts": ["C"], "reference": "R",
                }], "v1")

    def make_guard(self, validator_class):
        guard = step4.Guard()
        guard.configure(allow_metrics_collection=False)
        step4.settings.rc.enable_metrics = False
        return guard.use(validator_class(on_fail=step4.OnFailAction.FIX))

    def test_pii_guard_fixes_all_four_types_and_preserves_clean_text(self):
        guard = self.make_guard(step4.PIIDetector)
        text = "alice@example.com; (555) 867-5309; 123-45-6789; 4532 1234 5678 9010"
        with redirect_stdout(StringIO()):
            output = guard.validate(text).validated_output
        self.assertEqual(output, "[EMAIL_REDACTED]; [PHONE_REDACTED]; [SSN_REDACTED]; [CREDIT_CARD_REDACTED]")
        self.assertEqual(guard.validate("No PII here.").validated_output, "No PII here.")

    def test_json_guard_repairs_fences_quotes_commas_and_returns_fallback(self):
        guard = self.make_guard(step4.JSONFormatter)
        cases = [
            ('```json\n{"name": "Bob"}\n```', {"name": "Bob"}),
            ("{'name': 'Charlie', 'score': 95}", {"name": "Charlie", "score": 95}),
            ('{"items": [1, 2,],}', {"items": [1, 2]}),
            ('{"note": "it\'s fine ,}",}', {"note": "it's fine ,}"}),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw), redirect_stdout(StringIO()):
                self.assertEqual(json.loads(guard.validate(raw).validated_output), expected)
        valid = '{"name":"Alice"}'
        self.assertEqual(guard.validate(valid).validated_output, valid)
        fallback = json.loads(guard.validate("This is not JSON: {]").validated_output)
        self.assertIn("error", fallback)
        self.assertEqual(fallback["raw"], "This is not JSON: {]")


if __name__ == "__main__":
    unittest.main()
