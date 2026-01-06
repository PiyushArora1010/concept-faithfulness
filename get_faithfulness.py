import argparse

from tasks.faithfulness_engine import FaithfulnessEngine

def parse_args():
    parser = argparse.ArgumentParser()
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

if __name__ == '__main__':
    args = parse_args()
    engine = FaithfulnessEngine(args)
    
    tasks = args.task.split(",")
    tasks = [task.strip() for task in tasks]
    for task in tasks:
        print(f"Running task: {task}")
        engine.run(task)