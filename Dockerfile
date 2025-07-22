FROM nvcr.io/nvidia/pytorch:24.06-py3

# 필수 툴만 최소 설치
RUN apt-get update && apt-get install -y \
    git \
    nano \
    curl \
    wget \
    tree \
    && apt-get clean

# 최신 Huggingface 패키지만 추가
RUN pip install --upgrade pip && \
    pip install \
        transformers \
        datasets \
        peft \
        trl \
        accelerate \
        einops \
        scipy \
        wandb \
        tiktoken \
        sentencepiece \
        safetensors

# PYTHONPATH 환경변수 설정
ENV PYTHONPATH=/workspace

# 캐시 정리
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
