#!/bin/bash
ml CUDA/12.1
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate walkthetalk

export CUDA_VISIBLE_DEVICES=1
MODEL_TAG="Qwen/Qwen3-4B"

python3 run_train.py --dataset ethics --dataset_path "data/e-SNLI" \
--counterfactual_data_path "results/concept_outputs/ethics/Qwen3_32B/counterfactual_results.jsonl" --model_tag $MODEL_TAG \
--model_temperature 1.0 --model_max_tokens 1024 --model_batch_size 16 --implied_model_tag Qwen/Qwen3-8B \
--learning_rate 1e-5 --weight_decay 1e-3 --epochs 2 --gradient_accumulation_steps 2 --completions_per_prompt 16 --seed 0 \
--lora --lora_rank 32 --hint_cf --lora_layers q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
--eval_steps 500 --save_steps 100 \
--run_name "${MODEL_TAG}_ethics_grpo" --output_dir "${MODEL_TAG}_ethics_grpo" \