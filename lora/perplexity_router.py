# lora.perplexity_router.py
# 입력 문장의 perplexity 계산 및 그에 따른 LoRA rank 선택

def compute_perplexity(prompt, model, tokenizer):
    import torch
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()
    
"""
단순 룰 기반 LoRA rank 선택 함수
추후 이 자리에 MLP-based router 대체 가능
"""
def training_ranks(ppl):
    if ppl < 5:
        return []  # 쉬운 입력은 학습 생략
    elif ppl < 15:
        return [2]
    elif ppl < 25:
        return [2, 4]  # mismatch 방지 (15~25)
    elif ppl < 35:
        return [4]
    elif ppl <= 45:
        return [4, 8]  # mismatch 방지 (35~45)
    else:
        return [8]

def inference_rank(ppl):
    if ppl < 20: return 2  # 20 이하 → r2
    elif ppl < 40: return 4  # 20~40 → r4
    else: return 8  # 40 이상 → r8
