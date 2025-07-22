# train_tta_lora.py
# Perplexity 기반으로 입력마다 LoRA rank를 선택하여 학습하는 구조 (Test-Time Adaptive Learning)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
import os
from dataset_loader import load_squad
from perplexity_router import compute_perplexity, choose_lora_rank

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

# 2. LoRA PEFT 모델로 초기 래핑 (최대 rank 하나만 get_peft_model)
print("Applying initial LoRA adapter...")
config = LoraConfig(
    r=rank_options[-1],  # r=8
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, config)

# 3. 나머지 adapter 추가
print("Loading additional adapters...")
for rank in rank_options[:-1]:  # r=2, r=4
    config = LoraConfig(
        r=rank,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    adapter_name = f"r{rank}"
    model.add_adapter(adapter_name, config)

model.print_trainable_parameters()
model.train()

# 4. 데이터 로딩
encoded = load_squad(tokenizer, split="train", limit=max_samples)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = DataLoader(encoded, batch_size=1, shuffle=True, collate_fn=collator)

# 5. Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 6. 학습 루프
print("\nStarting adaptive training...\n")
for step, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].to(model.device)
    labels = batch["labels"].to(model.device)

    # Prompt decode → PPL 평가 → rank 선택
    prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    ppl = compute_perplexity(prompt, model, tokenizer)
    rank = choose_lora_rank(ppl)
    adapter_name = f"r{rank}"

    model.set_adapter(adapter_name)

    # Forward/backward
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


    print(f"[Step {step}] PPL={ppl:.2f}, rank={rank}, loss={loss.item():.4f}")

# 7. 저장
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
