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
def training_rank(ppl):
    if ppl < 10:  # 학습 안함
        return 0
    elif ppl < 20:
        return 2
    elif ppl < 40:
        return 4
    else:
        return 8

def inference_rank(ppl):
    if ppl < 20:
        return 2
    elif ppl < 40:
        return 4
    else:
        return 8
