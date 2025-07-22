# model_download.py
# # LLM & Tokenizer 다운로드 및 로딩

from lora_setup import apply_lora

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "NousResearch/Llama-2-7b-hf"  # or meta-llama/Llama-2-7b-hf

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # or torch.float16
    device_map="auto"
)

# LoRA 적용
model = apply_lora(model, rank=8)