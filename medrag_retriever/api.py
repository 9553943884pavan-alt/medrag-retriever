# medrag_retriever/api.py (NEW)
from .chain import answer_question
from .prompts import NO_CONTEXT_RESPONSE

def answer_for_chat(question: str, translate_fn=None, target_language: str = "en") -> dict:
    """
    Wraps build_adaptive_pipeline's output into the exact response shape
    needed by a chat API — citations formatted, optional translation applied.

    translate_fn: an optional callable(text: str, target_language: str) -> str,
    injected by the caller (backend) so this package doesn't need to own
    translation model loading itself.
    """
    result = answer_question(question)   # from build_adaptive_pipeline, already in scope

    citations = []
    if not result["declined"] if "declined" in result else result["answer"] != NO_CONTEXT_RESPONSE:
        # NOTE: build_adaptive_pipeline currently returns contexts as plain strings (page_content),
        # not full metadata — this needs a small update (see below) to expose source/score per citation
        pass

    answer_text = result["answer"]
    if translate_fn is not None and target_language != "en":
        answer_text = translate_fn(answer_text, target_language)

    return {
        "answer": answer_text,
        "declined": result["answer"] == NO_CONTEXT_RESPONSE,
        "citations": citations,   # populated once chain.py exposes full metadata (see fix below)
        "config_used": result["config_used"]
    }