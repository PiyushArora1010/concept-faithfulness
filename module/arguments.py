import argparse

def faithfulness_args(extra_args=None):
    parser = argparse.ArgumentParser()

    # faithfulness defines its own module because it does NOT use common_args
    parser.add_argument('--module', type=str, default='faithfulness')

    parser.add_argument('--dataset', type=str, default='bbq')
    parser.add_argument('--dataset_path', type=str, default='data/bbq')

    parser.add_argument('--task', type=str, default='original')
    parser.add_argument('--mode', type=str, default='global')

    parser.add_argument('--example_indices', type=str, default="all")

    parser.add_argument('--responses_original_dir', type=str, default='model_responses')
    parser.add_argument('--responses_counterfactual_dir', type=str, default='model_responses')
    parser.add_argument('--implied_concepts_dir', type=str, default='output')
    parser.add_argument('--output_path', type=str, default='output')
    
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args(extra_args)


def common_args(module_name):
    parser = argparse.ArgumentParser(add_help=False)

    # ONLY HERE we define module name
    parser.add_argument('--module', type=str, default=module_name)

    # dataset
    parser.add_argument('--dataset', type=str, default='bbq')
    parser.add_argument('--dataset_path', type=str, default='data/bbq')

    # task
    parser.add_argument('--task', type=str, default='original')

    # model settings
    parser.add_argument('--model_tag', type=str, default='Qwen3_2B')
    parser.add_argument('--model_max_tokens', type=int, default=256)
    parser.add_argument('--model_thinking', action='store_true')
    parser.add_argument('--model_temperature', type=float, default=0)
    parser.add_argument('--model_batch_size', type=int, default=4)
    parser.add_argument('--load_in_4bit', action='store_true')

    # batching
    parser.add_argument('--example_batch_size', type=int, default=8)
    parser.add_argument('--example_indices', type=str, default="all")

    # directories
    parser.add_argument('--output_dir', type=str, default='output')
    
    # random seed
    parser.add_argument('--seed', type=int, default=0)

    return parser

def train_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_size', type=float, default=0.7)
    parser.add_argument('--val_size', type=float, default=0.15)
    
    # Prompt to get model response
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--add_instr', type=str, default=None)
    parser.add_argument('--knn_rank', action='store_true')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--few_shot_prompt_name', type=str, default='few_shot_cot_prompt')
    
    # Implied concepts settings
    parser.add_argument('--implied_concepts_base_prompt_name', type=str, default='implied_concepts_prompt')
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
    parser.add_argument('--loss_computed_on', type=str, default="explanation", choices=["decision", "explanation", "both"])
    parser.add_argument('--debug', action='store_true')
    
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
    parser.add_argument('--dataset', type=str, default='bbq')
    parser.add_argument('--dataset_path', type=str, default='data/bbq')
    parser.add_argument('--verify', type=str, default=None)
    parser.add_argument('--example_indices', type=str, default='all')
    parser.add_argument('--counterfactual_data_path', type=str, default="results/concept_outputs/bbq/Llama3.3_70B")
    parser.add_argument('--response_data_path', type=str, default="results/model_responses/bbq/Llama3.3_70B/Qwen3_4B")
    parser.add_argument('--max_examples', type=int, default=100000000)
    
    parser.add_argument('--seed', type=int, default=0)
    
    return parser.parse_args()

def sft_args(extra_args=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_size', type=float, default=0.7)
    
    parser.add_argument('--explanations_data_path', type=str, default='output')

    # Prompt to get model response
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--add_instr', type=str, default=None)
    parser.add_argument('--knn_rank', action='store_true')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--few_shot_prompt_name', type=str, default='few_shot_cot_prompt')

    # Model settings
    parser.add_argument('--model_tag', type=str, default='unsloth/Qwen3-4B')
    parser.add_argument('--model_max_tokens', type=int, default=2048)
    parser.add_argument('--model_batch_size', type=int, default=8)

    # Training settings
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    
    # Save and Logging settings
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='deleteme')
    parser.add_argument('--save_steps', type=int, default=25)
    parser.add_argument('--eval_steps', type=int, default=25)
    
    # LoRA settings
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_layers', type=str, nargs='+', default=None)
    
    # Data settings
    parser.add_argument('--dataset', type=str, default='bbq')
    parser.add_argument('--dataset_path', type=str, default='data/bbq')
    parser.add_argument('--example_indices', type=str, default='all')
    parser.add_argument('--max_examples', type=int, default=100000000)
    
    # Seed
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()

def concept_args(extra_args=None):
    base = common_args("concept_intervention")
    parser = argparse.ArgumentParser(parents=[base])

    parser.add_argument('--only_concept_removals', action='store_true') # Used for Removal Only setting (Not used in the paper)
    
    parser.add_argument('--concept_id_base_prompt_name', type=str, default='concept_id_prompt') # Used to identify concept IDs
    parser.add_argument('--concept_values_base_prompt_name', type=str, default='concept_values_prompt') # Used to get possible concept values
    
    parser.add_argument('--counterfactual_gen_base_prompt_name', type=str, default='counterfactual_gen_replacements_prompt') # Used to generate counterfactuals
    
    parser.add_argument('--exclude_question_in_prompt', action='store_true') # Whether to exclude the question when generating concept IDs and values (Not used in the paper)
    
    parser.add_argument('--include_unknown_concept_values', action='store_true') # Whether to include unknown concept values (Not used in the paper)

    return parser.parse_args(extra_args)

def verification_args(extra_args=None):
    base = common_args("verification")
    parser = argparse.ArgumentParser(parents=[base])

    parser.add_argument('--verification_question_base_prompt_name', type=str, default='intervention_question_verification_prompt')
    parser.add_argument('--verification_anschoices_base_prompt_name', type=str, default='intervention_anschoices_verification_prompt')
    
    return parser.parse_args(extra_args)

def response_args(extra_args=None):
    base = common_args("model_response")
    parser = argparse.ArgumentParser(parents=[base])

    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--few_shot_prompt_name', type=str, default='few_shot_cot_prompt')
    parser.add_argument('--add_instr', type=str, default=None)
    parser.add_argument('--knn_rank', action='store_true')
    parser.add_argument('--n_completions', type=int, default=50)
    parser.add_argument('--intervention_dir', type=str, default='output')

    return parser.parse_args(extra_args)


def implied_args(extra_args=None):
    base = common_args("implied_concepts")
    parser = argparse.ArgumentParser(parents=[base])

    parser.add_argument('--implied_concepts_base_prompt_name', type=str, default='implied_concepts_prompt')
    parser.add_argument('--n_completions', type=int, default=1)
    parser.add_argument('--response_dir', type=str, default='output')
    parser.add_argument('--intervention_dir', type=str, default='output')

    return parser.parse_args(extra_args)

def data_generation_args(extra_args=None):
    base = common_args("data_generation")
    parser = argparse.ArgumentParser(parents=[base])

    parser.add_argument('--response_dir', type=str, default='output')
    parser.add_argument('--intervention_dir', type=str, default='output')
    
    return parser.parse_args(extra_args)

def baseline_args(extra_args=None):
    base = common_args("baseline")
    parser = argparse.ArgumentParser(parents=[base])

    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--few_shot_prompt_name', type=str, default='few_shot_cot_prompt')
    parser.add_argument('--add_instr', type=str, default=None)
    parser.add_argument('--knn_rank', action='store_true')
    
    parser.add_argument('--response_dir', type=str, default='output')
    parser.add_argument('--max_samples', type=int, default=100000000)

    return parser.parse_args(extra_args)

argumentDic = {
    "faithfulness": faithfulness_args,
    "concept_intervention": concept_args,
    "model_response": response_args,
    "implied_concepts": implied_args,
    "verification": verification_args,
    "data_generation": data_generation_args,
    "baseline": baseline_args
}
