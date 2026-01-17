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

    # batching
    parser.add_argument('--example_batch_size', type=int, default=8)
    parser.add_argument('--example_indices', type=str, default="all")

    # directories
    parser.add_argument('--output_dir', type=str, default='output')

    return parser

def train_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_size', type=float, default=0.7)
    parser.add_argument('--val_size', type=float, default=0.15)
    
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--add_instr', type=str, default=None)
    parser.add_argument('--knn_rank', action='store_true')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--few_shot_prompt_name', type=str, default='few_shot_cot_prompt')
    
    parser.add_argument('--implied_concepts_base_prompt_name', type=str, default='implied_concepts_prompt')
    
    parser.add_argument('--model_tag', type=str, default='Qwen3_8B')
    parser.add_argument('--model_max_tokens', type=int, default=512)
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=16)
    
    parser.add_argument('--dataset', type=str, default='bbq')
    parser.add_argument('--dataset_path', type=str, default='data/bbq')
    parser.add_argument('--example_indices', type=str, default='all')
    parser.add_argument('--counterfactual_data_path', type=str, default=None)
    parser.add_argument('--response_data_path', type=str, default=None)
    parser.add_argument('--max_examples', type=int, default=100)
    
    return parser.parse_args()
    

def concept_args(extra_args=None):
    base = common_args("concept_intervention")
    parser = argparse.ArgumentParser(parents=[base])

    parser.add_argument('--only_concept_removals', action='store_true')
    parser.add_argument('--concept_id_base_prompt_name', type=str, default='concept_id_prompt')
    parser.add_argument('--concept_values_base_prompt_name', type=str, default='concept_values_prompt')
    parser.add_argument('--counterfactual_gen_base_prompt_name', type=str, default='counterfactual_gen_removals_prompt')
    parser.add_argument('--exclude_question_in_prompt', action='store_true')
    parser.add_argument('--include_unknown_concept_values', action='store_true')

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


argumentDic = {
    "faithfulness": faithfulness_args,
    "concept_intervention": concept_args,
    "model_response": response_args,
    "implied_concepts": implied_args
}
