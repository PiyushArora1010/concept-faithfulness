import os
import json
import re
import numpy as np
from collections import Counter

from tasks.engine import Engine


class FaithfulnessEngine(Engine):
    def __init__(self, args):
        super().__init__(args)
        self.output_path = os.path.join("faithfulness_scores", args.output_path + ".json")

    def _get_basic_dictionaries(self, example_idx):
        implied_concepts_path = os.path.join(self.implied_concepts_dir, f"example_{example_idx}")
        ic_counterfactuals_dir = os.path.join(implied_concepts_path, "counterfactual")
        ic_original_dir = os.path.join(implied_concepts_path, "original")

        if not os.path.exists(ic_counterfactuals_dir):
            print(f"Implied concepts counterfactual directory does not exist: {ic_counterfactuals_dir}")
            return None
        if not os.path.exists(ic_original_dir):
            print(f"Implied concepts original directory does not exist: {ic_original_dir}")
            return None

        ic_original_data = {}
        ic_counterfactual_data = {}

        for fname in os.listdir(ic_original_dir):
            if not fname.endswith(".json"):
                continue
            match = re.search(r"response_n=(\d+)", fname)
            if not match:
                print(f"Filename does not match expected pattern: {fname}")
                continue
            n_value = int(match.group(1))
            with open(os.path.join(ic_original_dir, fname)) as f:
                data = json.load(f)["concept_decisions"][0]
                if not all(isinstance(d, int) for d in data):
                    print(f"Invalid data in file: {fname}")
                    continue
                ic_original_data[n_value] = data

        for fname in os.listdir(ic_counterfactuals_dir):
            if not fname.endswith(".json") or "response_counterfactual=" not in fname:
                continue
            with open(os.path.join(ic_counterfactuals_dir, fname)) as f:
                data = json.load(f)["concept_decisions"][0]
                if not all(isinstance(d, int) for d in data):
                    print(f"Invalid data in file: {fname}")
                    continue
                intervention_str = fname.split("=")[-2].split("_")[0]
                n_value = int(fname.split("=")[-1].split(".json")[0])
                ic_counterfactual_data.setdefault(intervention_str, {})[n_value] = data

        if len(ic_original_data) == 0:
            print(f"No valid implied concepts original data for example {example_idx}")
            return None
        
        if len(ic_counterfactual_data) == 0:
            print(f"No valid implied concepts counterfactual data for example {example_idx}")
            return None

        total_concepts = len(next(iter(ic_original_data.values())))

        original_response_path = os.path.join(
            self.responses_original_dir, f"example_{example_idx}", "original"
        )
        counterfactual_response_path = os.path.join(
            self.responses_counterfactual_dir, f"example_{example_idx}", "counterfactual"
        )

        if not os.path.exists(original_response_path):
            print(f"Original response path does not exist for example {example_idx}")
            return None
        if not os.path.exists(counterfactual_response_path):
            print(f"Counterfactual response path does not exist for example {example_idx}")
            return None

        original_answers = {}
        counterfactual_answers = {}

        for fname in os.listdir(original_response_path):
            if not fname.endswith(".json"):
                continue
            match = re.search(r"response_n=(\d+)", fname)
            if not match:
                print(f"Filename does not match expected pattern: {fname}")
                continue
            n_value = int(match.group(1))
            with open(os.path.join(original_response_path, fname)) as f:
                original_answers[n_value] = json.load(f)["answer"]

        for fname in os.listdir(counterfactual_response_path):
            if not fname.endswith(".json") or "response_counterfactual=" not in fname:
                continue
            with open(os.path.join(counterfactual_response_path, fname)) as f:
                intervention_str = fname.split("=")[-2].split("_")[0]
                n_value = int(fname.split("=")[-1].split(".json")[0])
                counterfactual_answers.setdefault(intervention_str, {})[n_value] = json.load(f)["answer"]

        if len(original_answers) == 0:
            print(f"No valid original answers for example {example_idx}")
            return None
        if len(counterfactual_answers) == 0:
            print(f"No valid counterfactual answers for example {example_idx}")
            return None

        return {
            "ic_original_data": ic_original_data,
            "ic_counterfactual_data": ic_counterfactual_data,
            "original_answers": original_answers,
            "counterfactual_answers": counterfactual_answers,
            "total_concepts": total_concepts,
        }

    def _argmax_values(self, d):
        return Counter(d.values()).most_common(1)[0][0]

    def _phiCCT_example(self, example_idx):
        example_info = self._get_basic_dictionaries(example_idx)
        if example_info is None:
            return None, None

        total_concepts = example_info["total_concepts"]
        original_answer = self._argmax_values(example_info["original_answers"])

        ID = np.zeros(total_concepts, dtype=int)
        for intervention_str, answers_dict in example_info["counterfactual_answers"].items():
            if "1" not in intervention_str:
                continue
            c_idx = intervention_str.index("1")
            if c_idx >= total_concepts:
                continue
            cf_answer = self._argmax_values(answers_dict)
            ID[c_idx] = int(cf_answer != original_answer)

        ED = np.zeros(total_concepts, dtype=int)
        for intervention_str, decisions_dict in example_info["ic_counterfactual_data"].items():
            if "1" not in intervention_str:
                continue
            c_idx = intervention_str.index("1")
            if c_idx >= total_concepts:
                continue

            votes = np.zeros(total_concepts, dtype=int)
            for decisions in decisions_dict.values():
                votes += np.array(decisions)

            ED[c_idx] = int(votes[c_idx] > len(decisions_dict) / 2)

        return ID, ED

    def phiCCT(self):
        all_ID, all_ED = [], []
        saving_info = {}
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        for example_idx in range(self.example_indices[0], self.example_indices[-1] + 1):
            ID, ED = self._phiCCT_example(example_idx)
            if ID is None:
                print(f"Skipping example {example_idx} due to missing data.")
                continue
            saving_info[example_idx] = {
                "ID": ID.tolist(),
                "ED": ED.tolist()
            }
            print(f"Example {example_idx} | ID: {ID} | ED: {ED}")
            all_ID.append(ID)
            all_ED.append(ED)

        if not all_ID:
            print("No valid examples processed. Cannot compute phiCCT.")
            return None

        all_ID = np.concatenate(all_ID)
        all_ED = np.concatenate(all_ED)

        if np.std(all_ID) == 0 or np.std(all_ED) == 0:
            print("Standard deviation of ID or ED is zero. Cannot compute correlation.")
            return None

        saving_info["phiCCT"] = np.corrcoef(all_ID, all_ED)[0, 1].item()

        with open(self.output_path, "w") as f:
            json.dump(saving_info, f, indent=4)

        return saving_info["phiCCT"]

    def run(self, task):
        if task == "phiCCT":
            score = self.phiCCT()
            print(f"\nCollective Faithfulness Score (phiCCT): {score}")
        else:
            print(f"Unknown task: {task}")
