# train_lora.py
# Flexi-LoRA 학습: 단일 LoRA 모듈에 대해 입력 PPL에 따라 rank 조정 학습

import os
import time
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
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)  #! shuffle=False로 변경하니 너무 PPL이 비슷한게 연속으로 나옴

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
    step_start = time.perf_counter()

    input_ids = batch["input_ids"].to(model.device)
    labels = batch["labels"].to(model.device)

    # 1. PPL 계산
    t1 = time.perf_counter()
    with torch.no_grad():
        prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        ppl = compute_perplexity(prompt, model, tokenizer)
        rank = training_rank(ppl)
    t2 = time.perf_counter()

    if rank == 0:
        print(f"[Step {step}] PPL={ppl:6.2f} → Skip (PPL Time: {t2 - t1:.2f}s)")
        continue

    # 2. Forward + Backward
    t3 = time.perf_counter()
    outputs = model(input_ids=input_ids, labels=labels, rank=rank)
    loss = outputs.loss
    loss.backward()
    t4 = time.perf_counter()

    # 3. Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    t5 = time.perf_counter()

    # 전체 타이밍
    print(f"[Step {step}] PPL={ppl:6.2f} → Rank={rank} → Loss={loss.item():.4f} | "
          f"PPL Time: {t2 - t1:.2f}s | FW/BW: {t4 - t3:.2f}s | Optim: {t5 - t4:.2f}s | Total: {t5 - step_start:.2f}s")

# 6. 저장
print("\n[Save] Saving Flexi-LoRA adapter...")
model.save_pretrained(adapter_root)
print(f"[Done] Saved to {adapter_root}")
