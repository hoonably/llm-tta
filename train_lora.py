# train_lora.py
# Perplexity 기반으로 입력마다 LoRA rank를 선택하여 학습하는 구조 (Test-Time Adaptive Learning)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
import torch
from torch.utils.data import DataLoader
from peft import get_peft_model
import os
from data.dataset_loader import load_squad
from lora.perplexity_router import compute_perplexity, training_ranks
from lora.lora_setup import add_all_lora_adapters

# 설정
model_id = "NousResearch/Llama-2-7b-hf"
adapter_root = "./adapters"
rank_options = [2, 4, 8]
max_samples = 300

# 1. 모델 로딩
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. LoRA 다중 어댑터 추가
print("Adding multiple LoRA adapters...")
model = get_peft_model(base_model, peft_config=None)  # dummy wrap
model = add_all_lora_adapters(model, rank_options)
model.print_trainable_parameters()
model.train()

# 3. 데이터 로딩
encoded = load_squad(tokenizer, split="train", limit=max_samples)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = DataLoader(encoded, batch_size=1, shuffle=True, collate_fn=collator)

# 4. Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 5. 학습 루프
print("\nStarting adaptive training...\n")
for step, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].to(model.device)
    labels = batch["labels"].to(model.device)

    # Prompt decode → PPL 평가 → rank 선택
    prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    ppl = compute_perplexity(prompt, model, tokenizer)
    ranks = training_ranks(ppl)

    if not ranks:
        print(f"[Step {step}] PPL={ppl:.2f}, skipped due to low difficulty.")
        continue

    for rank in ranks:
        adapter_name = f"r{rank}"
        model.set_adapter(adapter_name)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"[Step {step}] PPL={ppl:.2f}, rank={rank}, loss={loss.item():.4f}")

# 6. 저장
print("\nSaving adapters...")
for rank in rank_options:
    adapter_name = f"r{rank}"
    if adapter_name in model.peft_config:
        model.set_adapter(adapter_name)
        save_path = os.path.join(adapter_root, adapter_name)
        model.save_pretrained(save_path)
        print(f"Adapter {adapter_name} saved to {save_path}")
    else:
        print(f"[Skip] Adapter {adapter_name} was never used → not saved.")