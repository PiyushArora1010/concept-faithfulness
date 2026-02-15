import os
from datetime import datetime
import time
import wandb
import argparse
import unsloth
from tasks.train_engine import TrainEngineConditionsGRPO, GRPOTrainer
from module.arguments import train_args
from module.utils import print0, set_seed
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

def reward_function_faithfulness(prompts, completions, **kwargs):
    global engine, model, tokenizer
    
    # breakpoint()
    
    example_indices = kwargs["example_id"]
    concepts_list = kwargs["concepts"]
    concepts_values_list = kwargs["concept_values"]
    counterfactual_prompts = kwargs["counterfactual_prompts"]
    counterfactual_intervention_strings = kwargs["counterfactual_intervention_strings"]
    
    # Get only unique counterfactual prompts
    counterfactual_prompts = [counterfactual_prompts[i] for i in range(0, len(completions), engine.completions_per_prompt)]
    
    # Get counterfactual answers (-1 if not found)
    counterfactual_answers = engine._get_counterfactual_answers(
        model, counterfactual_prompts, example_indices
    ) 

    counterfactual_answers = [ans for ans in counterfactual_answers for _ in range(engine.completions_per_prompt)] # repeat answers per completion

    # Get example answers and masks telling whether answer was found (-1 if not found)
    example_answers = engine._get_answers_from_responses(completions, example_indices)

    # Get Successful Interventions
    successful_interventions = [
        engine._get_successful_interventions(
            counterfactual_answers[i],
            counterfactual_intervention_strings[i],
            len(concepts_list[i]),
            example_answers[i]
        )
        for i in range(len(completions))
    ]

    # Get Implied Concepts and mask for each response
    implied_concepts_bool, implied_concepts_responses = engine._get_implied_concepts(
        completions,
        example_answers,
        concepts_list,
        concepts_values_list,
        example_indices,
    )

    # Faitfhulness reward computation
    rewards = [engine._reward_faithfulness_response(
        implied_concepts_bool[i],
        successful_interventions[i],
    ) for i in range(len(completions))]
    
    for ix, answer in enumerate(example_answers):
        if answer != -1:
            rewards[ix] += 0.1  # Reward for having an answer
    
    for ix, counterfactual_answer in enumerate(counterfactual_answers):
        # total -1 entries in counterfactual_answer (List of int)
        total_no_answer = counterfactual_answer.count(-1)
        total_concepts_len = max(len(concepts_list[ix]), 1)
        rewards[ix] += 0.1 * (total_concepts_len - total_no_answer) / total_concepts_len  # Reward for having counterfactual answers
    
    logging_dict = dict(
        {
            "completions": completions[0],
            "example_answers": example_answers[0],
            "counterfactual_answers": counterfactual_answers[0],
            "implied_conditions": implied_concepts_bool[0],
            "successful_interventions": successful_interventions[0],
            "rewards": rewards[0],
        }
    )
    
    for key, value in logging_dict.items():
        print0(f"{key}\n{value}")
    
    return rewards

if __name__ == '__main__':
    args = train_args()
    print0("Setting up training engine...")
    engine = TrainEngineConditionsGRPO(args)
    
    print0("Preparing model and datasets...")
    model, tokenizer = engine._get_model_and_tokenizer()

    vllm_sampling_params = SamplingParams(
        seed = args.seed,
        temperature = args.model_temperature,
        max_tokens = args.model_max_tokens
    )

    train_dataset, val_dataset = engine._prepare_datasets(tokenizer)
    
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
        use_vllm=True,
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
    trainer.train()
    # model.save_pretrained(os.path.join(engine.output_dir, "model"), tokenizer)