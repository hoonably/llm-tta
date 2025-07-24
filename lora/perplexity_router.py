# lora.perplexity_router.py
# 입력 문장의 perplexity 계산 및 그에 따른 LoRA rank 선택



# 이 과정에서 정답(Response) 는 전혀 사용되지 않음.
# 오직 입력 Prompt 자체의 자기 예측 난이도만 보는 것.
# 그래서 이게 Test-Time Input Difficulty로 쓰이는 것.
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

# TODO: 대부분 ppl이 낮게 나와서 임시로 수정함
def training_rank(ppl):
    if ppl < 4:  # 학습 안함
        return 0
    elif ppl < 7:
        return 2
    elif ppl < 10:
        return 4
    else:
        return 8

# def training_rank(ppl):
#     if ppl < 10:  # 학습 안함
#         return 0
#     elif ppl < 20:
#         return 2
#     elif ppl < 40:
#         return 4
#     else:
#         return 8

def inference_rank(ppl):
    if ppl < 20:
        return 2
    elif ppl < 40:
        return 4
    else:
        return 8
