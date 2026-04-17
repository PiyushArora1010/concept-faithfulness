import argparse
from tasks.concept_intervention_engine import ConceptInterventionEngine

parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--dataset', type=str, default='esnli')
parser.add_argument('--dataset_path', type=str, default='data/e-SNLI')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--sample_size', type=int, default=None, help="Number of examples")

# task
parser.add_argument('--task', type=str, default='concepts')

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

# prompt files
parser.add_argument('--concept_id_base_prompt_name', type=str, default='concept_ids') # Used to identify concept IDs
parser.add_argument('--concept_values_base_prompt_name', type=str, default='concept_values') # Used to get possible concept values
parser.add_argument('--counterfactual_gen_base_prompt_name', type=str, default='counterfactual_gen_prompt') # Used to generate counterfactuals


if __name__ == "__main__":
    args = parser.parse_args()
    engine = ConceptInterventionEngine(args)
    engine.run()