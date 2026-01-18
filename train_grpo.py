import os
import argparse
import unsloth
from tasks.train_engine import TrainEngine
from module.arguments import train_args
from module.utils import print0
from trl import GRPOConfig, GRPOTrainer

def reward_function_faithfulness(prompts, completions, **kwargs):
    global engine
    
    example_indices = kwargs["example_id"]
    intervention_dict_list = kwargs["intervention_dict"]
    original_answers = kwargs["original_answers"]
    concepts_list = kwargs["concepts"]
    concept_values_list = kwargs["concept_values"]
    
    answers, answers_mask = engine._get_answers_from_responses(completions, kwargs.get("example_id"))
    implied_concepts, implied_mask = engine._get_implied_concepts(
        completions,
        answers,
        concepts_list,
        concept_values_list,
        intervention_dict_list,
    )
    
    final_mask = (answers_mask & implied_mask)
    successful_interventions = engine._get_successful_interventions(answers, original_answers)
    rewards = engine._phiCCT(implied_concepts, successful_interventions, final_mask)
    return rewards

if __name__ == '__main__':
    args = train_args()
    print0("Setting up training engine...")
    engine = TrainEngine(args)
    
    print0("Preparing model and datasets...")
    model, tokenizer = engine._get_model_and_tokenizer()
    train_dataset, val_dataset, test_dataset = engine._prepare_datasets()
    
    print0(f"Train dataset size: {len(train_dataset)}")
    print0(f"Validation dataset size: {len(val_dataset)}")
    print0(f"Test dataset size: {len(test_dataset)}")

    max_prompt_length = args.model_max_tokens
    max_seq_length = 2 * args.model_max_tokens

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = GRPOConfig(
        learning_rate=args.learning_rate,  #5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=args.logging_steps,  #1,
        per_device_train_batch_size=args.model_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Increase to 4 for smoother training
        num_generations=args.completions_per_prompt,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        max_steps=args.steps,  #250,
        save_steps=args.save_steps,  #25,
        max_grad_norm=0.1,
        report_to="wandb",  # Can use Weights & Biases
        output_dir=args.output_dir,  # Directory to save results
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_function_faithfulness,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print0("Starting training...")
    trainer.train()