# data/dataset_loader.py

from datasets import load_dataset
from transformers import PreTrainedTokenizer

DATASET_CACHE = "./datasets"

def load_squad(tokenizer, split="train", limit=500):
    dataset = load_dataset("squad", cache_dir=DATASET_CACHE)[split].select(range(limit))

    def preprocess(example):
        return tokenizer(example["context"], truncation=True, padding="max_length", max_length=256)

    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return dataset

def load_gsm8k(tokenizer: PreTrainedTokenizer, split="train", limit=500):
    dataset = load_dataset("gsm8k", "main", cache_dir=DATASET_CACHE)[split].select(range(limit))

    def preprocess(example):
        return tokenizer(example["question"], truncation=True, padding="max_length", max_length=256)

    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return dataset
