import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace


def load_base_model_and_tokenizer(model_id: str = "microsoft/Phi-3.5-mini-instruct"):
    """Loads the quantized model + tokenizer ONCE — reused by both LLM wrappers below
    to avoid loading the 3.8B model twice into GPU memory."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model.generation_config.max_length = None   # clears the max_new_tokens/max_length warning
    return model, tokenizer


def build_llm_pair(model, tokenizer, temperature: float = 0.1):
    """
    Builds TWO ChatHuggingFace wrappers sharing the SAME underlying model weights —
    one configured for long-form (max_new_tokens=512), one for short-form (300).
    This avoids reloading the model twice while still allowing per-question
    max_new_tokens selection via the router.
    """
    def _build(max_new_tokens):
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=True, return_full_text=False
        )
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))

    llm_long = _build(max_new_tokens=512)
    llm_short = _build(max_new_tokens=256)
    return llm_long, llm_short