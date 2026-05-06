import os
import re
import json
import numpy as np
import time
import wandb
import argparse
import unsloth
from tasks.train_engine import TrainBinaryEngine

from module.utils import print0, set_seed
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

def reward_function_clipped(prompts, completions, **kwargs):
    reasoning_content = []
    for completion in completions:
        match = re.search(r'<reasoning>(.*?)</reasoning>', completion, re.DOTALL)
        if match:
            reasoning_content.append(match.group(1).strip())
        else:
            reasoning_content.append("")
    
    word_counts = [len(content.split()) for content in reasoning_content]
    
    # Clip: f(x) = min(1, x/128)
    clipped_rewards = [min(1.0, count / 128.0) for count in word_counts]
    
    return clipped_rewards

def reward_function_faithfulness(prompts, completions, **kwargs):
    global engine, model, tokenizer
    # breakpoint()
    example_indices = kwargs["example_id"]
    original_prompts = kwargs["original_prompt"]
    conditions = kwargs["condition"]
    questions = kwargs["question"]
    
    # Get only unique counterfactual prompts
    original_answers = engine._get_original_answers(
        model, original_prompts
    ) 
    # breakpoint()
    # Get example answers and masks telling whether answer was found (-1 if not found)
    example_answers = engine._get_answers_from_responses(completions)

    # Get Successful Interventions
    successful_interventions = engine._get_successful_interventions(
        example_answers,
        original_answers
    )

    # Get Implied Conditions and mask for each response
    implied_conditions_bool, implied_conditions_responses = engine._get_implied_conditions(
        completions,
        conditions,
        questions
    )

    # Faitfhulness reward computation
    rewards = [engine._reward_faithfulness_response(
        implied_conditions_bool[i],
        successful_interventions[i],
    ) for i in range(len(completions))]
    
    length_bonuses = reward_function_clipped(
        prompts, completions
    )
    
    for ix, length_bonus in enumerate(length_bonuses):
        rewards[ix] += 0.5 * length_bonus  # Add length bonus to faithfulness reward
    
    logging_dict = dict(
        {
            "completions": completions[0],
            "example_answers": example_answers[0],
            "original_answers": original_answers[0],
            "conditions": conditions[0],
            "implied_response": implied_conditions_responses[0],
            "implied_conditions": implied_conditions_bool[0],
            "successful_interventions": successful_interventions[0],
            "rewards": rewards[0],
        }
    )
    
    for key, value in logging_dict.items():
        print0(f"{key}\n{value}")

    return rewards



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
parser.add_argument('--learning_rate', type=float, default=5e-6)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--completions_per_prompt', type=int, default=6)

# Save and Logging settings
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='deleteme')
parser.add_argument('--logging_steps', type=int, default=1)
parser.add_argument('--save_steps', type=int, default=25)
parser.add_argument('--eval_steps', type=int, default=25)

# LoRA settings
parser.add_argument('--lora', action='store_true')
parser.add_argument('--lora_rank', type=int, default=16)
parser.add_argument('--lora_layers', type=str, nargs='+', default=None)

# Data settings
parser.add_argument('--dataset', type=str, default='esnli')
parser.add_argument('--dataset_path', type=str, default='data/e-SNLI')
parser.add_argument('--counterfactual_data_path', type=str, default="results/concept_outputs/bbq/Llama3.3_70B")
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--hint_cf', action='store_true')
parser.add_argument('--sample_size', type=int, default=None, help="Number of examples")
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
set_seed(args.seed)

if __name__ == '__main__':

    print0("Setting up training engine...")
    engine = TrainBinaryEngine(args)
    
    print0("Preparing model and datasets...")
    model, tokenizer = engine._get_model_and_tokenizer()

    vllm_sampling_params = SamplingParams(
        seed = args.seed,
        temperature = args.model_temperature,
        max_tokens = args.model_max_tokens
    )

    train_dataset, val_dataset = engine._prepare_datasets(tokenizer)
    # breakpoint()
    
    print0(f"Train dataset size: {len(train_dataset)}")
    print0(f"Validation dataset size: {len(val_dataset)}")

    max_prompt_length = args.model_max_tokens
    max_seq_length = 2 * args.model_max_tokens

    RUN_NAME = engine.run_name

    wandb.init(
        project="Faithfulness ISO",
        name=RUN_NAME,  
        config=vars(args), 
    )

    training_args = GRPOConfig(
        use_vllm=True,  # Set to True to use vLLM for training (if supported)
        vllm_sampling_params=vllm_sampling_params,
        
        learning_rate=args.learning_rate,  #5e-6,
        lr_scheduler_type="cosine",
        
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.05,
        warmup_ratio=0,

        logging_steps=args.logging_steps,  #1,
        
        temperature=args.model_temperature,  #0.7,
        
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_accumulation_steps = args.gradient_accumulation_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Increase to 4 for smoother training
        num_generations=args.completions_per_prompt,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        
        num_train_epochs=args.epochs,  #2,
        save_steps=args.save_steps,  #25,
        eval_steps=args.eval_steps,  #25,
        
        max_grad_norm=0.3,
        report_to="wandb",  # Can use Weights & Biases
        output_dir=engine.output_dir,  # Directory to save results
        
        run_name = RUN_NAME,
        
        seed = args.seed,
        data_seed = args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_function_faithfulness,
            # reward_function_formatting,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print0("Starting training...")
    trainer.train() # resume_from_checkpoint = True
    # model.save_pretrained(os.path.join(engine.output_dir, "model"), tokenizer)