"""Bước 1: FAISS RAG pipeline và 50 traces rag-query trên LangSmith."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config  # Đặt LANGCHAIN_* trước khi import LangChain.

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from qa_pairs import SAMPLE_QUESTIONS
from utils.data_loader import build_vectorstore, load_knowledge_base, split_text
from utils.llm_factory import get_embeddings, get_llm


def setup_vectorstore():
    """Đọc knowledge base, chia chunk 500/50 và xây FAISS index."""
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    print(f"📚 Đã chia thành {len(chunks)} chunks")
    return build_vectorstore(chunks, embeddings)


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời. "
     "Trả lời bằng ngôn ngữ của câu hỏi. Nếu context không đủ, hãy nói rõ "
     "không có đủ thông tin.\n\nContext:\n{context}"),
    ("human", "{question}"),
])


def build_rag_chain(vectorstore):
    """Nối retriever → prompt → LLM → parser; trả về (chain, retriever)."""
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )
    return chain, retriever


@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """Ghi trace cha, các bước LCEL tự tạo run con có input/output."""
    return chain.invoke(question)


def main():
    print("=" * 60)
    print("  Bước 1: LangSmith RAG Pipeline — LeVanSang-2A202602391")
    print("=" * 60)
    if not config.validate():
        sys.exit(1)
    vectorstore = setup_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question}")
        print(f"       A: {answer}\n")
    print(f"✅ Đã xử lý {len(SAMPLE_QUESTIONS)} câu hỏi.")
    print(f"   Kiểm tra ≥ 50 traces 'rag-query' trong project '{config.LANGSMITH_PROJECT}'.")
    print("   Mở https://smith.langchain.com và xác nhận retriever, context, answer.")


if __name__ == "__main__":
    main()
