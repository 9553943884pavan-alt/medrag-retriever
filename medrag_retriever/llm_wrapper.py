import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM , pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import BitsAndBytesConfig


def build_llm(model_id: str = "microsoft/Phi-3.5-mini-instruct", max_new_tokens: int = 300, temperature: float = 0.1):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
                                  )
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto",quantization_config=bnb_config)

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
