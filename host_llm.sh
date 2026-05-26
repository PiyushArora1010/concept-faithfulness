#!/bin/bash

ml tools/prod
ml miniforge
ml CUDA/12.1

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate walkthetalk

cd $PBS_O_WORKDIR
export CUDA_VISIBLE_DEVICES=0

MODEL=$1

python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --port 3316

# --quantization bitsandbytes \
# --load-format bitsandbytes \