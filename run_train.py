import os
import json
import numpy as np
import time
import wandb
import argparse
import unsloth
from tasks.train_engine import TrainEngine

from module.utils import print0, set_seed
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

def reward_function_faithfulness(prompts, completions, **kwargs):
    global engine, model, tokenizer

    example_indices = kwargs["example_id"]
    counterfactual_prompts = kwargs["counterfactual_prompts"]
    original_questions = kwargs["question"]
    conditions_list = kwargs["original_conditions"]
    
    # Get only unique counterfactual prompts
    counterfactual_prompts = [counterfactual_prompts[i] for i in range(0, len(completions), engine.completions_per_prompt)]
    counterfactual_answers = engine._get_counterfactual_answers(
        model, counterfactual_prompts
    ) 
    counterfactual_answers = [ans for ans in counterfactual_answers for _ in range(engine.completions_per_prompt)] # repeat answers per completion

    # Get example answers and masks telling whether answer was found (-1 if not found)
    example_answers = engine._get_answers_from_responses(completions)

    # Get Successful Interventions
    successful_interventions = [
        engine._get_successful_interventions(
            counterfactual_answers[i],
            example_answers[i]
        )
        for i in range(len(completions))
    ]

    # Get Implied Conditions and mask for each response
    implied_conditions_bool, implied_conditions_responses = engine._get_implied_conditions(
        completions,
        conditions_list,
        original_questions,
    )

    # Faithfulness reward computation
    rewards = [engine._reward_faithfulness_response(
        implied_conditions_bool[i],
        successful_interventions[i],
    ) for i in range(len(completions))]
    
    # for ix, answer in enumerate(example_answers):
    #     if answer != -1:
    #         rewards[ix] += 0.1  # Reward for having an answer
    
    logging_dict = dict(
        {
            "completions": completions[0],
            "example_answers": example_answers[0],
            "counterfactual_answers": counterfactual_answers[0],
            "implied_conditions": implied_conditions_bool[0],
            "successful_interventions": successful_interventions[0],
            "rewards": rewards[0],
        }
    )
    
    for key, value in logging_dict.items():
        print0(f"{key}\n{value}")

    return rewards

def reward_correct_answer(prompts, completions, **kwargs):
    global engine, model, tokenizer

    gts = kwargs["gt"]

    example_answers = engine._get_answers_from_responses(completions)

    print0(f"Example answers: {example_answers}")
    print0(f"Ground truth answers: {gts}")

    rewards = []    
    for i in range(len(completions)):
        if example_answers[i] == gts[i]:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

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
parser.add_argument('--dataset', type=str, default='ethics')
parser.add_argument('--dataset_path', type=str, default='data/e-SNLI')
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
            reward_correct_answer,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print0("Starting training...")
    trainer.train()
    # model.save_pretrained(os.path.join(engine.output_dir, "model"), tokenizer)