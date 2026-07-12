from langchain_core.output_parsers import StrOutputParser
from .prompts import NO_CONTEXT_RESPONSE
from .router import get_adaptive_config
from .retriever import HybridMedRAGRetriever


def format_docs(docs) -> str:
    if not docs:
        return ""
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)


def build_adaptive_pipeline(client, dense_model, sparse_model, reranker, llm_long, llm_short,
                             collection_name: str = "medrag_textbooks", top_k_fetch: int = 20):
    """
    Returns a single callable that, given a question, picks the right config
    (long vs short) and runs retrieval + generation accordingly.
    """

    def answer_question(question: str) -> dict:
        config = get_adaptive_config(question)

        # FIXED: real arguments instead of "...", pulled from the enclosing
        # build_adaptive_pipeline scope (client, dense_model, etc.) plus the
        # config-specific values (top_n_final, min_rerank_score) from the router
        retriever = HybridMedRAGRetriever(
            client=client,
            dense_model=dense_model,
            sparse_model=sparse_model,
            reranker=reranker,
            collection_name=collection_name,
            top_k_fetch=top_k_fetch,
            top_n_final=config.top_n_final,
            min_rerank_score=config.min_rerank_score
        )
        docs = retriever.invoke(question)

        if not docs:
            return {"answer": NO_CONTEXT_RESPONSE, "citations": [], "config_used": config.label}

        context_str = format_docs(docs)
        llm_to_use = llm_long if config.label == "long" else llm_short
        chat_chain = config.prompt_template | llm_to_use | StrOutputParser()
        answer = chat_chain.invoke({"context": context_str, "question": question})

        citations = [
            {
                "source": d.metadata["source"],
                "chunk_id": d.metadata["chunk_id"],
                "excerpt": d.page_content[:200],
                "full_text": d.page_content,
                "rerank_score": d.metadata["rerank_score"]
            }
            for d in docs
        ]

        return {"answer": answer, "citations": citations, "config_used": config.label}

    return answer_question
