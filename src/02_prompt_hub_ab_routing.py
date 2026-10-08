"""Bước 2: push/pull Prompt Hub và A/B routing tất định bằng MD5."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config  # Phải import trước LangChain.

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client, traceable
from langsmith.utils import LangSmithConflictError

from qa_pairs import SAMPLE_QUESTIONS
from utils.data_loader import build_vectorstore, load_knowledge_base, split_text
from utils.evidence import tee_logs
from utils.llm_factory import get_embeddings, get_llm


PROMPT_V1_NAME = "le-van-sang-2a202602391-rag-prompt-v1"
PROMPT_V2_NAME = "le-van-sang-2a202602391-rag-prompt-v2"

# V1 trả lời trực tiếp; V2 giải thích có cấu trúc và điều kiện áp dụng.
# Bước 3 dùng chính xác hai chuỗi này.
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


def push_prompts_to_hub(client: Client):
    """Đẩy hai prompt lên Hub, phân biệt lỗi thật với Nothing to commit."""
    for version, name, prompt, description in [
        ("V1", PROMPT_V1_NAME, PROMPT_V1, "LeVanSang-2A202602391: trả lời ngắn gọn"),
        ("V2", PROMPT_V2_NAME, PROMPT_V2, "LeVanSang-2A202602391: giải thích có cấu trúc"),
    ]:
        try:
            url = client.push_prompt(name, object=prompt, description=description)
            print(f"✅ Đã push {version} → {url}")
        except LangSmithConflictError as exc:
            if "nothing to commit" not in str(exc).lower():
                raise
            print(f"ℹ️ {version} đã có trên Hub, nội dung không đổi (Nothing to commit).")


def pull_prompts_from_hub(client: Client, *, allow_fallback: bool = True) -> dict:
    """Có nhánh fallback để debug; main yêu cầu pull thành công từ Hub."""
    prompts = {}
    for name, local in [(PROMPT_V1_NAME, PROMPT_V1), (PROMPT_V2_NAME, PROMPT_V2)]:
        try:
            prompt = client.pull_prompt(name)
            if not isinstance(prompt, ChatPromptTemplate):
                raise TypeError(f"Prompt '{name}' không phải ChatPromptTemplate")
            if set(prompt.input_variables) != {"context", "question"}:
                raise ValueError(f"Prompt '{name}' phải nhận context và question")
            prompts[name] = prompt
            print(f"↓ Đã pull '{name}' từ Hub")
        except Exception:
            if not allow_fallback:
                raise
            prompts[name] = local
            print(f"⚠️ Dùng prompt local (fallback) cho '{name}'; chưa đạt tiêu chí Hub.")
    return prompts


def get_prompt_version(request_id: str) -> str:
    """Cùng request_id luôn cho cùng tên prompt: MD5 chẵn → V1, lẻ → V2."""
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """Retrieve trong trace cha và trả lời bằng prompt đã pull từ Hub."""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context, "question": question,
    })
    return {"question": question, "answer": answer, "version": version}


def setup_vectorstore():
    """Dùng lại đúng dữ liệu, chunking 500/50 và FAISS của bước 1."""
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    return build_vectorstore(chunks, embeddings)


def _run():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing — LeVanSang-2A202602391")
    print("=" * 60)
    if not config.validate():
        sys.exit(1)
    client = Client(api_key=config.LANGSMITH_API_KEY)
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client, allow_fallback=False)
    vectorstore = setup_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    v1_count, v2_count = 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]
        result = ask_ab(retriever, llm, prompt, question, version_tag)
        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}/50] [{request_id}] [prompt-{version_tag}] {question}")
        print(f"       A: {result['answer']}\n")
    print(f"📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    print("✅ Bước 2 hoàn thành. Xác nhận 2 prompt trên Hub và ≥ 100 traces trong project.")
    client.flush()


def main():
    with tee_logs("02_ab_routing_log.txt"):
        _run()


if __name__ == "__main__":
    main()
