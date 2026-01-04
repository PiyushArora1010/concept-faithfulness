import os
import json
import re
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tasks.engine import Engine

class FaithfulnessEngine(Engine):
    def __init__(self, args):
        super().__init__(args)
        self.output_path = Path("faithfulness_scores") / f"{args.output_path}.json"
        self.num_workers = getattr(args, 'num_workers', os.cpu_count())

    def _load_ic_data(self, directory: Path, is_counterfactual: bool = False) -> Dict:
        result = {}
        
        for fpath in directory.glob("*.json"):
            if is_counterfactual and "response_counterfactual=" not in fpath.name:
                continue
            
            try:
                with fpath.open() as f:
                    data = json.load(f)["concept_decisions"][0]
                    if not all(isinstance(d, int) for d in data):
                        continue
                    
                    if is_counterfactual:
                        parts = fpath.stem.split("=")
                        intervention_str = parts[-2].split("_")[0]
                        n_value = int(parts[-1])
                        result.setdefault(intervention_str, {})[n_value] = data
                    else:
                        match = re.search(r"response_n=(\d+)", fpath.name)
                        if match:
                            n_value = int(match.group(1))
                            result[n_value] = data
            except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
                print(f"Error processing {fpath.name}: {e}")
                continue
        
        return result

    def _load_answers(self, directory: Path, is_counterfactual: bool = False) -> Dict:
        result = {}
        
        for fpath in directory.glob("*.json"):
            if is_counterfactual and "response_counterfactual=" not in fpath.name:
                continue
            
            try:
                with fpath.open() as f:
                    answer = json.load(f)["answer"]
                    
                    if is_counterfactual:
                        parts = fpath.stem.split("=")
                        intervention_str = parts[-2].split("_")[0]
                        n_value = int(parts[-1])
                        result.setdefault(intervention_str, {})[n_value] = answer
                    else:
                        match = re.search(r"response_n=(\d+)", fpath.name)
                        if match:
                            n_value = int(match.group(1))
                            result[n_value] = answer
            except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
                print(f"Error processing {fpath.name}: {e}")
                continue
        
        return result

    def _get_basic_dictionaries(self, example_idx: int) -> Optional[Dict]:
        base_path = Path(self.implied_concepts_dir) / f"example_{example_idx}"
        ic_cf_dir = base_path / "counterfactual"
        ic_orig_dir = base_path / "original"
        
        resp_base = Path(self.responses_original_dir) / f"example_{example_idx}"
        orig_resp_path = resp_base / "original"
        
        cf_resp_base = Path(self.responses_counterfactual_dir) / f"example_{example_idx}"
        cf_resp_path = cf_resp_base / "counterfactual"

        required_paths = [
            (ic_orig_dir, "Implied concepts original directory"),
            (ic_cf_dir, "Implied concepts counterfactual directory"),
            (orig_resp_path, "Original response path"),
            (cf_resp_path, "Counterfactual response path")
        ]
        
        for path, description in required_paths:
            if not path.exists():
                print(f"{description} does not exist: {path}")
                return None

        ic_original_data = self._load_ic_data(ic_orig_dir, is_counterfactual=False)
        ic_counterfactual_data = self._load_ic_data(ic_cf_dir, is_counterfactual=True)
        original_answers = self._load_answers(orig_resp_path, is_counterfactual=False)
        counterfactual_answers = self._load_answers(cf_resp_path, is_counterfactual=True)

        if not all([ic_original_data, ic_counterfactual_data, original_answers, counterfactual_answers]):
            print(f"Missing data for example {example_idx}")
            return None

        return {
            "ic_original_data": ic_original_data,
            "ic_counterfactual_data": ic_counterfactual_data,
            "original_answers": original_answers,
            "counterfactual_answers": counterfactual_answers,
        }

    @staticmethod
    def _argmax_values(d: Dict) -> any:
        return Counter(d.values()).most_common(1)[0][0]

    def _phiCCT_example(self, example_idx: int) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
        example_info = self._get_basic_dictionaries(example_idx)
        if example_info is None:
            return example_idx, None, None

        original_answer = self._argmax_values(example_info["original_answers"])

        ID = {}
        for intervention_str, answers_dict in example_info["counterfactual_answers"].items():
            c_idx = intervention_str.find("1")
            if c_idx == -1:
                continue
            cf_answer = self._argmax_values(answers_dict)
            ID[c_idx] = int(cf_answer != original_answer)

        ED = {}
        for intervention_str, decisions_dict in example_info["ic_counterfactual_data"].items():
            c_idx = intervention_str.find("1")
            if c_idx == -1:
                continue

            votes = Counter()
            for decisions in decisions_dict.values():
                votes.update(dict(enumerate(decisions)))

            ED[c_idx] = int(votes[c_idx] > len(decisions_dict) / 2)

        common_indices = sorted(set(ID.keys()) & set(ED.keys()))
        if not common_indices:
            return example_idx, None, None
        
        ID_array = np.array([ID[i] for i in common_indices])
        ED_array = np.array([ED[i] for i in common_indices])
        
        return example_idx, ID_array, ED_array

    def phiCCT(self) -> Optional[float]:
        all_ID, all_ED = [], []
        saving_info = {}
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        example_indices = list(range(self.example_indices[0], self.example_indices[-1] + 1))
        total_examples = len(example_indices)
        batch_size = max(self.num_workers * 2, 50)
        
        print(f"Processing {total_examples} examples with {self.num_workers} workers in batches of {batch_size}")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for batch_start in range(0, total_examples, batch_size):
                batch_end = min(batch_start + batch_size, total_examples)
                batch_indices = example_indices[batch_start:batch_end]
                
                print(f"Processing batch {batch_start//batch_size + 1}/{(total_examples + batch_size - 1)//batch_size} "
                      f"(examples {batch_indices[0]}-{batch_indices[-1]})")
                
                futures = {executor.submit(self._phiCCT_example, idx): idx for idx in batch_indices}
                
                for future in as_completed(futures):
                    example_idx, ID, ED = future.result()
                    
                    if ID is None or ED is None:
                        print(f"Skipping example {example_idx} due to missing data.")
                        continue
                    
                    saving_info[example_idx] = {"ID": ID.tolist(), "ED": ED.tolist()}
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

        saving_info["phiCCT"] = float(np.corrcoef(all_ID, all_ED)[0, 1])

        with self.output_path.open("w") as f:
            json.dump(saving_info, f, indent=4)

        print(f"\nphiCCT: {saving_info['phiCCT']:.4f} (detailed results saved to {self.output_path})")
        return saving_info["phiCCT"]

    def run(self, task: str):
        if task == "phiCCT":
            score = self.phiCCT()
        else:
            print(f"Unknown task: {task}")