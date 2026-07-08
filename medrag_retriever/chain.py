from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .prompts import generation_prompt, NO_CONTEXT_RESPONSE


def format_docs(docs) -> str:
    if not docs:
        return ""
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)


def build_chain(hybrid_retriever, llm):
    def route_on_docs(inputs: dict):
        if not inputs["docs"]:
            return NO_CONTEXT_RESPONSE
        chain_inputs = {"context": format_docs(inputs["docs"]), "question": inputs["question"]}
        chat_chain = generation_prompt | llm | StrOutputParser()
        return chat_chain.invoke(chain_inputs)

    return (
        {"docs": hybrid_retriever, "question": RunnablePassthrough()}
        | RunnableLambda(route_on_docs)
    )