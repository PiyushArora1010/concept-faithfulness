import argparse

def faithfulness_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, default='faithfulness', help='module name')
    
    parser.add_argument('--dataset', type=str, default='bbq', help='dataset name')
    parser.add_argument('--dataset_path', type=str, default='data/bbq', help='path to dataset')

    parser.add_argument('--task', type=str, default='original', help='task to perform')
    parser.add_argument('--mode', type=str, default='global', help='mode: global or mean')

    parser.add_argument('--example_indices', type=str, default="all", help='list of example indices to process')

    parser.add_argument('--responses_original_dir', type=str, default='model_responses', help='directory for original responses')
    parser.add_argument('--responses_counterfactual_dir', type=str, default='model_responses', help='directory for counterfactual responses')
    parser.add_argument('--implied_concepts_dir', type=str, default='output', help='output directory')
    parser.add_argument('--output_path', type=str, default='output', help='output directory')
    
    return parser.parse_args()

def common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, default='common', help='module name')
    
    parser.add_argument('--dataset', type=str, default='bbq', help='dataset name')
    parser.add_argument('--dataset_path', type=str, default='data/bbq', help='path to dataset')

    parser.add_argument('--model_tag', type=str, default='Qwen3_2B', help='name of model to use to generate counterfactuals data')
    parser.add_argument('--model_max_tokens', type=int, default=256, help='max tokens for LLM-based counterfactual generation model. Only relevant for completion GPT (since default max tokens is inf for Chat GPT).')
    parser.add_argument('--model_thinking', action='store_true', help='whether to enable "thinking" mode for Qwen models')
    parser.add_argument('--model_temperature', type=float, default=0, help='temperature for language model used for counterfactual example generation steps')
    parser.add_argument('--model_batch_size', type=int, default=4, help='batch size for LLM model inference')
    
    parser.add_argument('--example_batch_size', type=int, default=8, help='batch size for processing examples')
    parser.add_argument('--example_indices', type=str, default="all", help='list of example indices to process')

    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    return parser

def concept_args():
    parser = common_args()
    parser.add_argument('--only_concept_removals', action='store_true', help='whether to only include concept removals in the intervention generation')

    parser.add_argument('--concept_id_base_prompt_name', type=str, default='concept_id_prompt', help='name of base prompt for concept ID step')
    parser.add_argument('--concept_values_base_prompt_name', type=str, default='concept_values_prompt', help='name of base prompt for concept values ID step')
    parser.add_argument('--counterfactual_gen_base_prompt_name', type=str, default='counterfactual_gen_removals_prompt', help='name of base prompt for counterfactual generation step')
    parser.add_argument('--include_unknown_concept_values', action='store_true', help='whether to include unknown as a concept value in the intervention generation')
    
    return parser.parse_args()

def response_args():
    parser = common_args()
    parser.add_argument('--cot', action='store_true', help='whether to use chain-of-thought prompting')
    parser.add_argument('--few_shot', action='store_true', help='whether to use few-shot prompting')
    parser.add_argument('--few_shot_prompt_name', type=str, default='few_shot_cot_prompt', help='name of chain-of-thought prompt to use')
    parser.add_argument('--n_completions', type=int, default=50, help='number of completions to generate per example')

    parser.add_argument('--intervention_dir', type=str, default='output', help='output directory')
    
    return parser.parse_args()

def implied_args():
    parser = common_args()
    parser.add_argument('--implied_concepts_base_prompt_name', type=str, default='implied_concepts_prompt', help='name of base prompt for implied concepts identification step')
    parser.add_argument('--n_completions', type=int, default=1, help='number of completions to generate per example')

    parser.add_argument('--response_dir', type=str, default='output', help='output directory')
    parser.add_argument('--intervention_dir', type=str, default='output', help='output directory')
    
    return parser.parse_args()


argumentDic = {
    "faithfulness": faithfulness_args,
    "concept_intervention": concept_args,
    "model_response": response_args,
    "implied_concepts": implied_args
}