import os
import wandb
import argparse
import unsloth
from tasks.train_engine import TrainEngineGRPO
from module.arguments import train_args
from module.utils import print0, set_seed
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

def reward_function_faithfulness(prompts, completions, **kwargs):
    global engine, model, tokenizer
    example_indices = kwargs["example_id"]
    intervention_dict_list = kwargs["intervention_dict"]
    original_prompts = kwargs["original_prompt"]
    concepts_list = kwargs["concepts"]
    concept_values_list = kwargs["concept_values"]
    
    answers = engine._get_answers_from_responses(completions, kwargs.get("example_id"))
 
    implied_concepts, implied_concepts_responses = engine._get_implied_concepts(
        completions,
        answers,
        concepts_list,
        concept_values_list,
        intervention_dict_list,
    )
    
    original_prompts = [original_prompts[i] for i in range(0, len(completions), engine.completions_per_prompt)]
    original_answers, original_responses = engine._get_original_answers(
        model,
        original_prompts,
        example_indices
    )
    original_answers = [ans for ans in original_answers for _ in range(engine.completions_per_prompt)] # repeat answers per completion
    
    successful_interventions = engine._get_successful_interventions(answers, original_answers)

    rewards = engine._phiCCT(implied_concepts, successful_interventions)
    
    for ix, answer in enumerate(answers):
        if answer != -1:
            rewards[ix] += 0.1  # Reward for having an answer
    
    for ix, original_answer in enumerate(original_answers):
        if original_answer != -1:
            rewards[ix] += 0.1  # Reward for having an original answer
    
    logging_dict = {
        "Example ID": example_indices[0],
        "CounterFactual Response": completions[0],
        "CounterFactual Answer": answers[0],
        "Original Response": original_responses[0],
        "Original Answer": original_answers[0],
        
        "Intervention String": intervention_dict_list[0]["intervention_str"],
        "Implied Concept?": implied_concepts[0],
        "Successful Intervention?": successful_interventions[0],
        "Reward": rewards[0],
    }
    for key, value in logging_dict.items():
        print0(f"{key}: {value}")
        print0("-"*20)
 
    return rewards

def reward_function_formatting(prompts, completions, **kwargs):
    global engine
    rewards = []
    for completion in completions:
        index_answer = engine.dataset.answer_starting_index(completion, engine.prompting_strategy)
        if index_answer == -1:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards

if __name__ == '__main__':
    args = train_args()
    print0("Setting up training engine...")
    engine = TrainEngineGRPO(args)
    
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
        eval_dataset=val_dataset
    )
    
    print0("Starting training...")
    trainer.train()