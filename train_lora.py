# train_lora.py
# Flexi-LoRA 학습: 단일 LoRA 모듈에 대해 입력 PPL에 따라 rank 조정 학습

import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from data.dataset_loader import load_squad
from lora.perplexity_router import compute_perplexity, training_rank
from lora.lora_setup import patch_lora_layers

# 설정
model_id = "NousResearch/Llama-2-7b-hf"
cache_dir = "./checkpoints/llama-2-7b"
adapter_root = "./adapters"
batch_size = 1
max_samples = 300
lr = 5e-5
r_max = 8  # 최대 rank

# 1. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token

# 2. 데이터셋
dataset = load_squad(tokenizer, split="train", limit=max_samples)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

# 3. 모델 및 Flexi-LoRA 초기화
print("Loading base model & applying Flexi-LoRA...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=cache_dir,
)
patch_lora_layers(base_model, r_max=r_max)
model = base_model
model.train()

# 4. Optimizer (LoRA param만)
lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
optimizer = torch.optim.AdamW(lora_params, lr=lr)

# 5. 학습 루프
print("\n[Flexi-LoRA Training Start]\n")
for step, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].to(model.device)
    labels = batch["labels"].to(model.device)

    with torch.no_grad():
        prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        ppl = compute_perplexity(prompt, model, tokenizer)
        rank = training_rank(ppl)

    if rank == 0:
        print(f"[Step {step}] PPL={ppl:6.2f} → Skip")
        continue  # 쉬운 샘플은 생략

    outputs = model(input_ids=input_ids, labels=labels, rank=rank)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"[Step {step}] PPL={ppl:6.2f} → Rank={rank} → Loss={loss.item():.4f}")

# 6. 저장
print("\n[Save] Saving Flexi-LoRA adapter...")
model.save_pretrained(adapter_root)
print(f"[Done] Saved to {adapter_root}")
