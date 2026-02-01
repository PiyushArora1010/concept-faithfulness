import argparse
from module.arguments import argumentDic

def parse_args():
    top_parser = argparse.ArgumentParser(add_help=False)
    top_parser.add_argument('--module', type=str, default='faithfulness', choices=["faithfulness", "concept_intervention", "verification", "model_response", "implied_concepts", "data_generation", "baseline"])
    
    args, remaining_argv = top_parser.parse_known_args()

    if args.module in argumentDic:
        return argumentDic[args.module](remaining_argv)  # pass remaining args
    else:
        raise ValueError(f"Unknown module {args.module}")


if __name__ == '__main__':
    args = parse_args()

    # Instantiate engine
    if args.module == "faithfulness":
        from tasks.faithfulness_engine import FaithfulnessEngine
        engine = FaithfulnessEngine(args)
    elif args.module == "concept_intervention":
        from tasks.concept_intervention_engine import ConceptInterventionEngine
        engine = ConceptInterventionEngine(args)
    elif args.module == "verification":
        from tasks.verification_engine import VerificationEngine
        engine = VerificationEngine(args)
    elif args.module == "model_response":
        from tasks.model_response_engine import ModelResponseEngine
        engine = ModelResponseEngine(args)
    elif args.module == "implied_concepts":
        from tasks.implied_concepts_engine import ImpliedConceptsEngine
        engine = ImpliedConceptsEngine(args)
    elif args.module == "data_generation":
        from tasks.data_generation_engine import DataGenerationEngine
        engine = DataGenerationEngine(args)
    elif args.module == "baseline":
        from tasks.baseline_engine import BaselineEngine
        engine = BaselineEngine(args)
    else:
        raise ValueError(f"Unknown module {args.module}")

    # 3. Handle "task" only if it exists
    if hasattr(args, "task"):
        tasks = [t.strip() for t in args.task.split(",")]
    else:
        tasks = ["default"]

    for task in tasks:
        print(f"Running task: {task}")
        engine.run(task)
