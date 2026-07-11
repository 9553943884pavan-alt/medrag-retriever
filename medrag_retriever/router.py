from dataclasses import dataclass
from .prompts import long_form_prompt_template, short_form_prompt_template


@dataclass
class RetrievalGenConfig:
    """Bundles all settings that should vary together based on question complexity."""
    prompt_template: object
    top_n_final: int
    max_new_tokens: int
    min_rerank_score: float
    label: str   # for logging/analysis — tracks which path was used


def get_adaptive_config(question_text: str, word_threshold: int = 40) -> RetrievalGenConfig:
    """
    Routes to a 'long/complex question' config (more context, more reasoning room)
    or a 'short/direct question' config (leaner, faster) based on question length.

    Empirically grounded in project findings:
    - Long clinical vignettes (MedQA-style, ~177 avg tokens) benefit from top_n_final=5
      and full chain-of-thought reasoning space (max_new_tokens=512).
    - Short factual questions (MMLU-style, ~63 avg tokens) perform comparably with
      top_n_final=3 and max_new_tokens=300, at meaningfully lower latency.
    - min_rerank_score=0.125 is held constant across both paths — diagnostics showed
      retrieval threshold is not the binding constraint; the LLM's own self-check
      behavior is, which the prompt variants address instead.
    """
    word_count = len(question_text.split())
    is_long_question = word_count > word_threshold

    if is_long_question:
        return RetrievalGenConfig(
            prompt_template=long_form_prompt_template,
            top_n_final=5,
            max_new_tokens=512,
            min_rerank_score=0.125,
            label="long"
        )
    else:
        return RetrievalGenConfig(
            prompt_template=short_form_prompt_template,
            top_n_final=3,
            max_new_tokens=300,
            min_rerank_score=0.125,
            label="short"
        )