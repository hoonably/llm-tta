# run_inference.py
# 입력에 따라 PPL 기반으로 LoRA adapter 선택 → 답변 생성 (TTA)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from data.dataset_loader import load_squad
from lora.perplexity_router import compute_perplexity, inference_rank
import os
from lora.lora_setup import add_all_lora_adapters

model_id = "NousResearch/Llama-2-7b-hf"
adapter_root = "./adapters"  # r2, r4, r8 저장된 경로

# 1. Load base model/tokenizer
print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. Initialize PEFT wrapper and add adapters
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, os.path.join(adapter_root, "r8"))
model.load_adapter(os.path.join(adapter_root, "r2"), adapter_name="r2")
model.load_adapter(os.path.join(adapter_root, "r4"), adapter_name="r4")
model.load_adapter(os.path.join(adapter_root, "r8"), adapter_name="r8")

# 3. Load dataset
print("Loading evaluation dataset...")
dataset = load_squad(tokenizer, split="validation", limit=10)

# 4. Inference loop with adapter switching
print("\nRunning adaptive inference...\n")
for i, example in enumerate(dataset):
    input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(model.device)
    prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # Compute PPL & select rank
    ppl = compute_perplexity(prompt, model, tokenizer)
    rank = inference_rank(ppl)
    adapter_name = f"r{rank}"
    model.set_adapter(adapter_name)

    # Generate response
    outputs = model.generate(input_ids, max_new_tokens=50)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"[{i}] PPL: {ppl:.2f} → Rank {rank} → Adapter: {adapter_name}")
    print(f"Prompt: {prompt[:80]}...")
    print(f"Output: {output_text[len(prompt):].strip()}")
    print("="*60)
