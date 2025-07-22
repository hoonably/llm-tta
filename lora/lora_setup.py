# lora/lora_setup.py
# LoRA 단일 랭크 어댑터 적용 함수

from peft import LoraConfig, get_peft_model

def apply_lora(model, rank=8):
    """
    단일 rank의 LoRA 어댑터를 HuggingFace 모델에 적용
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
    return model
