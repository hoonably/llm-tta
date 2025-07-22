#!/bin/bash

# bash run_docker.sh -> 인자에 아무것도 넣지 않으면 device==all
# bash run_docker.sh 1 3 -> 인자에 하나 이상 띄어쓰기로 넣으면 device==1,3 이런 식으로 됨

if [ $# -eq 0 ]; then
  GPU_DEVICES=all
else
  GPU_DEVICES=$(echo "$@" | tr ' ' ',')
fi

echo "Using GPU(s): $GPU_DEVICES"

sudo docker run --gpus "\"device=$GPU_DEVICES\"" -it --rm \
  --name llm-tta-container \
  --workdir /workspace \
  -v "$PWD":/workspace \
  llm-tta bash
