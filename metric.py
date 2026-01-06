import argparse

from tasks.faithfulness_engine import FaithfulnessEngine
from tasks.concept_intervention_engine import ConceptInterventionEngine
from tasks.model_response_engine import ModelResponseEngine
from tasks.implied_concepts_engine import ImpliedConceptsEngine

from module.arguments import argumentDic

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, default='faithfulness', choices=["faithfulness", "concept_intervention", "model_response", "implied_concepts"], help='module to run')
    args, unknown = parser.parse_known_args()
    
    if args.module in argumentDic:
        return argumentDic[args.module]()
    else:
        raise ValueError(f"Unknown module: {args.module}")
    
if __name__ == '__main__':
    args = parse_args()
    
    if args.module == "faithfulness":
        engine = FaithfulnessEngine(args)
    elif args.module == "concept_intervention":
        engine = ConceptInterventionEngine(args)
    elif args.module == "model_response":
        engine = ModelResponseEngine(args)
    elif args.module == "implied_concepts":
        engine = ImpliedConceptsEngine(args)
    else:
        raise ValueError(f"Unknown module: {args.module}")
    
    tasks = args.task.split(",")
    tasks = [task.strip() for task in tasks]
    for task in tasks:
        print(f"Running task: {task}")
        engine.run(task)
    