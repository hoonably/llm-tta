# llm-tta

Parameter-Efficient Fine-Tuning

### LLM + Test-Time + 난이도 기반 LoRA rank 선택

Pretrained LLM에 대해 Test-Time LoRA를 수행

입력 난이도에 따라 rank를 동적으로 조절하여 **빠르면서도 성능 손실이 적은** LoRA fine-tuning

```
    ┌────────────────────────────┐
    │     Pretrained LLM         │
    └────────────┬───────────────┘
                 │
    ┌────────────▼──────────────┐
    │     Token Embedding       │
    └────────────┬──────────────┘
                 │
         ┌───────▼────────┐
         │   Rank Router  │   ◀──── (Perplexity 기반 또는 학습된 MLP)
         └───────┬────────┘
                 │ rank ∈ {2, 4, 8}
    ┌────────────▼──────────────┐
    │   LoRA Adapter (slice)    │   ◀──── (선택된 rank에 해당하는 A_r, B_r)
    └────────────┬──────────────┘
                 │
        ┌────────▼────────┐
        │   LLM Layer i   │
        └─────────────────┘
```

