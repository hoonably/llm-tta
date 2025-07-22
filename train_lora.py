# train_lora.py
# Flexi-LoRA 기반: 하나의 shared LoRA 모듈 + PPL 기반 rank 선택 학습

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch
import os

from data.dataset_loader import load_squad
from lora.perplexity_router import compute_perplexity, training_rank
from lora.lora_setup import patch_lora_layers

# 설정
model_id = "NousResearch/Llama-2-7b-hf"
cache_dir = "./checkpoints/llama-2-7b"
adapter_root = "./adapters"
rank_options = [2, 4, 8]
r_max = max(rank_options)
max_samples = 300
lr = 5e-5
batch_size = 1

# 1. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token

# 2. 데이터셋
dataset = load_squad(tokenizer, split="train", limit=max_samples)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

# 3. 모델 + Flexi-LoRA patch
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=cache_dir
)
patch_lora_layers(base_model, target_modules=["q_proj", "v_proj"], r_max=r_max)
model = base_model.train()

# 4. Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# 5. 학습 루프
print("\n[Flexi-LoRA Training Start]\n")
for step, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].to(model.device)
    labels = batch["labels"].to(model.device)

    # ① PPL 측정
    prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    ppl = compute_perplexity(prompt, model, tokenizer)

    # ② rank 결정
    rank = training_rank(ppl)
    if rank == 0:  # 학습 안함
        print(f"[Step {step}] PPL={ppl:.2f}, rank={rank}, skipping step.")
        continue

    outputs = model(input_ids=input_ids, labels=labels, rank=rank)

    # ③ forward (rank 인자로 전달됨)
    outputs = model(input_ids=input_ids, labels=labels, rank=rank)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"[Step {step}] PPL={ppl:.2f}, rank={rank}, loss={loss.item():.4f}")

# 6. 저장 (전체 모델 저장 or Lora weight만 저장 가능)
save_path = os.path.join(adapter_root, "flexi-lora")
model.save_pretrained(save_path)
print(f"\n[Saved] Flexi-LoRA model saved to {save_path}")
