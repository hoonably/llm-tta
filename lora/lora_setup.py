# lora_setup.py
# LoRA 구조 래핑 및 동적 rank 대응 코드

from peft import LoraConfig, get_peft_model

def apply_lora(model, rank=8):
    """
    단일 LoRA adapter 구성용.
    Baseline 또는 단일 rank 실험 시 사용.
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

def add_all_lora_adapters(model, rank_list=[2, 4, 8]):
    """
    다중 LoRA adapter (예: r2, r4, r8) 추가용.
    Test-Time Adaptive Learning에 사용.
    """
    for rank in rank_list:
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
    return model
