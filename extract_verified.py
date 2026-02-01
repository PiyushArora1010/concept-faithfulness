import os
import glob
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--concepts_dir", type=str, required=True, help="Directory containing concept JSON files")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save results")
parser.add_argument("--verify_tag", type=str, default="Qwen3_32B", help="Tag indicating verified concepts")
args = parser.parse_args()

def get_verified_concepts(example_id):
    example_dir = os.path.join(args.concepts_dir, f"example_{example_id}")
    concept_path = os.path.join(example_dir, "concepts.json")
    concept_settings_path = os.path.join(example_dir, "concept_settings.json")
    if not os.path.exists(concept_path):
        return None, None, None
    with open(concept_path, "r") as f:
        concepts = json.load(f)
    
    if not os.path.exists(concept_settings_path):
        return None, None, None
    with open(concept_settings_path, "r") as f:
        concept_settings = json.load(f)
        
    verification_file = os.path.join(example_dir, f"verification_concepts_{args.verify_tag}.json")
    if not os.path.exists(verification_file):
        return None, None, None
    
    with open(verification_file, "r") as f:
        verification_data = json.load(f)
        verification_bool = [item["verification"] == "YES" for item in verification_data]
        
    verified_concepts = [concept for concept, verified in zip(concepts, verification_bool) if verified]
    verified_settings = [setting for setting, verified in zip(concept_settings, verification_bool) if verified]
    
    return verification_bool, verified_concepts, verified_settings

def get_verified_counterfactuals(example_id, verification_concepts_bool):
    example_dir = os.path.join(args.concepts_dir, f"example_{example_id}")
    counterfactual_files = glob.glob(os.path.join(example_dir, "counterfactual_*1*.json"))

    if not counterfactual_files:
        return None
    
    verified_counterfactual_data = []
    
    for counterfactual_file in counterfactual_files:
        with open(counterfactual_file, "r") as f:
            counterfactual_data = json.load(f)
            
        verification_counterfactual_file = counterfactual_file.replace("counterfactual", f"verification_counterfactual_{args.verify_tag}")
        if not os.path.exists(verification_counterfactual_file):
            continue
        
        with open(verification_counterfactual_file, "r") as f:
            verification_data = json.load(f)
            verification_bool = verification_data["verification"] == "YES"
            
        if not verification_bool:
            continue
        
        counterfactual_data["intervention_str"] = "".join([i for i, v in zip(counterfactual_data["intervention_str"], verification_concepts_bool) if v])
        if "1" not in counterfactual_data["intervention_str"]:
            continue
        
        counterfactual_data["old_values"] = [val for val, v in zip(counterfactual_data["old_values"], verification_concepts_bool) if v]
        counterfactual_data["new_values"] = [val for val, v in zip(counterfactual_data["new_values"], verification_concepts_bool) if v]
        
        verified_counterfactual_data.append(counterfactual_data)
        
    return verified_counterfactual_data

if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True)
    example_indices = 2600
    
    for example_id in range(example_indices):
        verified_concepts_path = os.path.join(args.output_dir, f"example_{example_id}", "concepts.json")
        verified_settings_path = os.path.join(args.output_dir, f"example_{example_id}", "concept_settings.json")
        
        if os.path.exists(os.path.join(args.output_dir, f"example_{example_id}")):
            print(f"Example {example_id} already processed. Skipping.")
            continue

        verification_concepts_bool, verified_concepts, verified_settings = get_verified_concepts(example_id)
        
        if verification_concepts_bool is None:
            print(f"No concept verification data for example {example_id}. Skipping.")
            continue
        
        verified_counterfactuals = get_verified_counterfactuals(example_id, verification_concepts_bool)
        
        
        os.makedirs(os.path.dirname(verified_concepts_path), exist_ok=True)
        with open(verified_concepts_path, "w") as f:
            json.dump(verified_concepts, f, indent=4)
            
        with open(verified_settings_path, "w") as f:
            json.dump(verified_settings, f, indent=4)
            
        for counterfactual_data in verified_counterfactuals:
            intervention_str = counterfactual_data["intervention_str"]
            counterfactual_path = os.path.join(args.output_dir, f"example_{example_id}", f"counterfactual_{intervention_str}.json")
            with open(counterfactual_path, "w") as f:
                json.dump(counterfactual_data, f, indent=4)
        
        print(f"Processed example {example_id}: {len(verified_concepts)} verified concepts, {len(verified_counterfactuals)} verified counterfactuals.")