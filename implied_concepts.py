import argparse

from tasks.implied_concepts_engine import ImpliedConceptsEngine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bbq', help='dataset name')
    parser.add_argument('--dataset_path', type=str, default='data/bbq', help='path to dataset')

    parser.add_argument('--task', type=str, default='original', help='task to perform')

    parser.add_argument('--model_tag', type=str, default='Qwen3_2B', help='name of model to use to generate counterfactuals data')
    parser.add_argument('--model_max_tokens', type=int, default=256, help='max tokens for LLM-based counterfactual generation model. Only relevant for completion GPT (since default max tokens is inf for Chat GPT).')
    parser.add_argument('--model_thinking', action='store_true', help='whether to enable "thinking" mode for Qwen models')
    parser.add_argument('--model_temperature', type=float, default=0, help='temperature for language model used for counterfactual example generation steps')
    parser.add_argument('--model_batch_size', type=int, default=4, help='batch size for LLM model inference')
    
    parser.add_argument('--implied_concepts_base_prompt_name', type=str, default='implied_concepts_prompt', help='name of base prompt for implied concepts identification step')
    parser.add_argument('--n_completions', type=int, default=1, help='number of completions to generate per example')
    
    parser.add_argument('--example_batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--example_indices', type=str, default="all", help='list of example indices to process')

    parser.add_argument('--response_dir', type=str, default='output', help='output directory')
    parser.add_argument('--intervention_dir', type=str, default='output', help='output directory')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    engine = ModelResponseEngine(args)
    
    tasks = args.task.split(",")
    tasks = [task.strip() for task in tasks]
    for task in tasks:
        print(f"Running task: {task}")
        engine.run(task)