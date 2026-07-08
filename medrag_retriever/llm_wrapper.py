import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM , pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace


def build_llm(model_id: str = "microsoft/Phi-3.5-mini-instruct", max_new_tokens: int = 512, temperature: float = 0.1):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        return_full_text=False
    )
    base_llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return ChatHuggingFace(llm=base_llm)