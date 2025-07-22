# run_inference.py
# 입력 난이도에 따라 PPL 기반으로 LoRA adapter 선택 → 답변 생성 (TTA)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data.dataset_loader import load_squad
from lora.perplexity_router import compute_perplexity, inference_rank
import torch
import os

model_id = "NousResearch/Llama-2-7b-hf"
cache_dir = "./checkpoints/llama-2-7b"
adapter_root = "./adapters"

# 1. Load base model/tokenizer
print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=cache_dir
)

# 2. Load evaluation dataset
print("Loading evaluation dataset...")
dataset = load_squad(tokenizer, split="validation", limit=10)

# 3. Inference loop with dynamic LoRA loading
print("\nRunning adaptive inference...\n")
for i, example in enumerate(dataset):
    input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(base_model.device)
    prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # Compute PPL and choose rank
    ppl = compute_perplexity(prompt, base_model, tokenizer)
    rank = inference_rank(ppl)
    adapter_path = os.path.join(adapter_root, f"r{rank}")

    # Load adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # Generate response
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=50)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"[{i}] PPL: {ppl:.2f} → Rank {rank} → Adapter: r{rank}")
    print(f"Prompt: {prompt[:80]}...")
    print(f"Output: {output_text[len(prompt):].strip()}")
    print("="*60)
