import os
import re
import json
import numpy as np
import time
import wandb
import argparse
import unsloth
from tasks.dpo_engine import TrainEngine

from module.utils import print0, set_seed
from vllm import SamplingParams

parser = argparse.ArgumentParser()

# Implied concepts settings
parser.add_argument('--implied_model_thinking', action='store_true')
parser.add_argument('--implied_model_url', type=str, default="http://localhost:3316/v1")
parser.add_argument('--implied_model_tag', type=str, default="Qwen/Qwen3-4B")
parser.add_argument('--implied_model_max_tokens', type=int, default=2048)

# Model settings
parser.add_argument('--model_tag', type=str, default='unsloth/Qwen3-4B')
parser.add_argument('--model_max_tokens', type=int, default=2048)
parser.add_argument('--model_temperature', type=float, default=1)
parser.add_argument('--model_batch_size', type=int, default=8)
parser.add_argument('--model_thinking', action='store_true')

# Training settings
parser.add_argument('--rounds', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=5e-6)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--completions_per_prompt', type=int, default=16)

# Save and Logging settings
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='deleteme')

# LoRA settings
parser.add_argument('--lora', action='store_true')
parser.add_argument('--lora_rank', type=int, default=16)
parser.add_argument('--lora_layers', type=str, nargs='+', default=None)

# Data settings
parser.add_argument('--dataset', type=str, default='ethics')
parser.add_argument('--dataset_path', type=str, default='data/e-SNLI')
parser.add_argument('--context_cf', action='store_true')
parser.add_argument('--hint_cf', action='store_true')
parser.add_argument('--counterfactual_data_path', type=str, default="results/concept_outputs/bbq/Llama3.3_70B")
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--sample_size', type=int, default=None, help="Number of examples")
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
set_seed(args.seed)

if __name__ == '__main__':

    print0("Setting up training engine...")
    engine = TrainEngine(args)
    
    wandb.init(
        project="Faithfulness ISO",
        name=args.run_name,  
        config=vars(args), 
    )

    engine.run()