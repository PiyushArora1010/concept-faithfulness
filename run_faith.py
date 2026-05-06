import argparse
from tasks.faith_engine import FaithEngine

parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--dataset', type=str, default='esnli')
parser.add_argument('--dataset_path', type=str, default='data/e-SNLI')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--sample_size', type=int, default=None, help="Number of examples")

# task
parser.add_argument('--task', type=str, default='original_responses')
parser.add_argument('--hint_types', type=str, default=None)

# model settings
parser.add_argument('--model_tag', type=str, default='Qwen3_2B')
parser.add_argument('--model_max_tokens', type=int, default=256)
parser.add_argument('--model_thinking', action='store_true')
parser.add_argument('--model_top_p', type=float, default=None)
parser.add_argument('--model_top_k', type=int, default=None)
parser.add_argument('--model_min_p', type=float, default=None)
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


if __name__ == "__main__":
    args = parser.parse_args()
    engine = FaithEngine(args)
    engine.run()