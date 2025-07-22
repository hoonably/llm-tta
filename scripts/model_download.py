# scripts/model_download.py
# 모델과 토크나이저 다운로드 및 rank별 LoRA adapter skeleton 저장

from transformers import AutoModelForCausalLM, AutoTokenizer
from lora.lora_setup import apply_lora
import torch
import os

model_id = "NousResearch/Llama-2-7b-hf"
cache_dir = "./checkpoints/llama-2-7b"
adapter_root = "./adapters"
rank_list = [2, 4, 8]

# tokenizer는 고정
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

# 각 rank별로 base model → LoRA 적용 → 저장
for rank in rank_list:
    print(f"\n[Download] Applying adapter for rank {rank}...")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir
    )

    model = apply_lora(base_model, rank=rank)
    save_path = os.path.join(adapter_root, f"r{rank}")
    model.save_pretrained(save_path)

    print(f"[Save] LoRA adapter for rank {rank} saved to {save_path}")
