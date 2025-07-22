# lora_setup.py
# LoRA 구조 래핑 및 동적 rank 대응 코드

from peft import LoraConfig, get_peft_model

def apply_lora(model, rank=8):
    """
    model: HuggingFace Transformers LLM
    rank: int, 최대 LoRA rank (최대치까지 전체 weight 구성)
    return: LoRA-wrapped model
    """
    config = LoraConfig(
        r=rank,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model