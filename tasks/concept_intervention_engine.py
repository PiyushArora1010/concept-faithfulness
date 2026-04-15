import os
import re
import json

from tasks.engine import Engine

class ConceptInterventionEngine(Engine):
    def __init__(self, args):
        super().__init__(args)
        self.output_dir = os.path.join(
            "results", "concept_outputs", self.dataset_tag, args.output_dir
        )
        self._get_model()
        
        self.concept_results_file = "concept_results.json"
        with open(os.path.join("prompts", self.dataset_tag, f"{self.concept_id_base_prompt_name}.txt"), 'r') as f:
            self.concept_id_base_prompt = f.read()
            
        with open(os.path.join("prompts", self.dataset_tag, f"{self.concept_values_base_prompt_name}.txt"), 'r') as f:
            self.concept_values_base_prompt = f.read()
        
        self.counterfactual_results_file = "counterfactual_results.json"
        with open(os.path.join("prompts", self.dataset_tag, f"{self.counterfactual_gen_base_prompt_name}.txt"), 'r') as f:
            self.counterfactual_gen_base_prompt = f.read()

    def parse_concept_values(self, response):
        pattern = r'^\s*\d+\.\s*\(A\)\s*(.*?)\s*\(B\)\s*(.*)$'

        current_values = []
        alternative_values = []

        for line in response.split("\n"):
            match = re.match(pattern, line)
            if match:
                current = match.group(1).strip()
                alternative = match.group(2).strip()

                current_values.append(current)
                alternative_values.append(alternative)

        return current_values, alternative_values

    def _get_concept_values_batch(self, example_indices):
        prompts = []
        for cnt, example_idx in enumerate(example_indices):
            prompt = self.concept_values_base_prompt + "\n" + self.dataset[example_idx]["question"]
            prompts.append(prompt)
            
        if len(prompts) == 0:
            return

        responses = self.model.batch_generate_response(prompts)
        
        batched_concept_values = {}
        for cnt, example_idx in enumerate(example_indices):
            response = responses[cnt]
            print(self.dataset[example_idx]["question"], end="\n")
            print(f"LLM Response for concept settings for example {example_idx}:\n{response}\n")
            
            try:
                original_values, alternaltive_values = self.parse_concept_values(response)
                
                assert len(original_values) == len(alternaltive_values), "Number of original values does not match number of alternative value sets."
                # assert len(original_values) == len(batched_concept_ids[example_idx]), "Number of concept settings does not match number of concepts."
                batched_concept_values[example_idx] = {
                    "original_values": original_values,
                    "alternative_values": alternaltive_values
                }
                print(f"Parsed concept settings for example {example_idx}:\nOriginal values: {original_values}\nAlternative values: {alternaltive_values}\n")
            except Exception as e:
                print(f"Error parsing concept settings for example {example_idx}: {e}. Saving response for analysis.")
                original_values, alternaltive_values = None, None
                
        return batched_concept_values

    def _get_concepts_and_values(self):
        batch_size = self.example_batch_size
        batch_counter = 0   
        example_indices_batch = []
        
        for example_idx in range(len(self.dataset)):
            if batch_counter >= batch_size:
                concept_values = self._get_concept_values_batch(example_indices_batch)
                
                with open(os.path.join(self.output_dir, self.concept_results_file), 'a') as f:
                    for idx in concept_values.keys():
                        result = {
                            "example_idx": idx,
                            # "concept_ids": concept_ids[idx],
                            "concept_values": concept_values[idx]
                        }
                        f.write(json.dumps(result) + "\n")
                
                example_indices_batch = []
                batch_counter = 0
            
            example_indices_batch.append(example_idx)
            batch_counter += 1
            
        if len(example_indices_batch) > 0:
            concept_values = self._get_concept_values_batch(example_indices_batch)
            
            with open(os.path.join(self.output_dir, self.concept_results_file), 'a') as f:
                for idx in concept_values.keys():
                    result = {
                        "example_idx": idx,
                        # "concept_ids": concept_ids[idx],
                        "concept_values": concept_values[idx]
                    }
                    f.write(json.dumps(result) + "\n")
    
    def parse_counterfactual(self, response):
        pattern_context = r'^Counterfactual Context:\s*(.+)$'
        pattern_question = r'^Counterfactual Question:\s*(.+)$'
        
        context = None
        question = None
        
        for line in response.split("\n"):
            match_context = re.match(pattern_context, line)
            if match_context:
                context = match_context.group(1).strip()
            match_question = re.match(pattern_question, line)
            if match_question:
                question = match_question.group(1).strip()
        
        return context, question

    def _get_counterfactuals_batch(self, example_indices, concept_values_dict):
        prompts = []
        metadata = []
        
        for example_idx in example_indices:
            original_context_question = self.dataset[example_idx]["question"]
            
            concept_vals = concept_values_dict.get(example_idx)
            if not concept_vals:
                print(f"No concept values for example {example_idx}")
                continue
            
            original_values = concept_vals["original_values"]
            alternative_values = concept_vals["alternative_values"]
            
            for cond_idx, (orig_val, alt_val) in enumerate(zip(original_values, alternative_values)):
                conditions_list = "\n".join([f"{i+1}. {v}" for i, v in enumerate(original_values)])
                target_line = f"Target condition: change condition {cond_idx+1} from \"{orig_val}\" to \"{alt_val}\""
                
                input_section = f"{original_context_question}\nExtracted conditions:\n{conditions_list}\n{target_line}"
                full_prompt = self.counterfactual_gen_base_prompt.strip() + "\n\n" + input_section
                prompts.append(full_prompt)
                metadata.append((example_idx, cond_idx, orig_val, alt_val))
        
        if not prompts:
            return []
        
        responses = self.model.batch_generate_response(prompts)
        
        # Group results by example_idx
        results_by_example = {}
        for cnt, (example_idx, cond_idx, orig_val, alt_val) in enumerate(metadata):
            response = responses[cnt]
            print(f"LLM Response for example {example_idx}, condition {cond_idx}:\n{response}\n")
            try:
                counterfactual_context, counterfactual_question = self.parse_counterfactual(response)
                if counterfactual_context is None or counterfactual_question is None:
                    raise ValueError("Could not parse both context and question")
                
                cf_entry = {
                    "condition_index": cond_idx,
                    "original_value": orig_val,
                    "alternative_value": alt_val,
                    "counterfactual_context": counterfactual_context,
                    "counterfactual_question": counterfactual_question
                }
                
                if example_idx not in results_by_example:
                    context_question = self.dataset[example_idx]["question"]
                    context = context_question.split("\nQuestion:")[0].replace("Context:", "").strip() if "\nQuestion:" in context_question else context_question.strip()
                    question = context_question.split("\nQuestion:")[1].strip() if "\nQuestion:" in context_question else ""
                    results_by_example[example_idx] = {
                        "example_idx": example_idx,
                        "original_context": context,
                        "original_question": question,
                        "counterfactuals": []
                    }
                results_by_example[example_idx]["counterfactuals"].append(cf_entry)
                print(f"Generated counterfactual for example {example_idx}, condition {cond_idx}")
            except Exception as e:
                print(f"Error parsing counterfactual for example {example_idx}, condition {cond_idx}: {e}")
                cf_entry = {
                    "condition_index": cond_idx,
                    "original_value": orig_val,
                    "alternative_value": alt_val,
                    "error": str(e),
                    "raw_response": response
                }
                if example_idx not in results_by_example:
                    context_question = self.dataset[example_idx]["question"]
                    context = context_question.split("\nQuestion:")[0].replace("Context:", "").strip() if "\nQuestion:" in context_question else context_question.strip()
                    question = context_question.split("\nQuestion:")[1].strip() if "\nQuestion:" in context_question else ""
                    results_by_example[example_idx] = {
                        "example_idx": example_idx,
                        "original_context": context,
                        "original_question": question,
                        "counterfactuals": []
                    }
                results_by_example[example_idx]["counterfactuals"].append(cf_entry)
        
        # Convert to list of example objects
        return list(results_by_example.values())

    def _generate_counterfactuals(self):
        concept_results_path = os.path.join(self.output_dir, self.concept_results_file)
        if not os.path.exists(concept_results_path):
            raise FileNotFoundError(f"Concept results not found at {concept_results_path}. Run concepts task first.")
        
        concept_values_dict = {}
        with open(concept_results_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                example_idx = data["example_idx"]
                concept_values_dict[example_idx] = data["concept_values"]
        
        batch_size = self.example_batch_size
        batch_counter = 0
        example_indices_batch = []
        
        all_example_indices = list(range(len(self.dataset)))
        available_indices = [idx for idx in all_example_indices if idx in concept_values_dict]
        
        # We'll collect results per batch and write them as nested JSON objects
        for example_idx in available_indices:
            if batch_counter >= batch_size:
                batch_results = self._get_counterfactuals_batch(example_indices_batch, concept_values_dict)
                with open(os.path.join(self.output_dir, self.counterfactual_results_file), 'a') as f:
                    for example_obj in batch_results:
                        f.write(json.dumps(example_obj) + "\n")
                example_indices_batch = []
                batch_counter = 0
            example_indices_batch.append(example_idx)
            batch_counter += 1
        
        if example_indices_batch:
            batch_results = self._get_counterfactuals_batch(example_indices_batch, concept_values_dict)
            with open(os.path.join(self.output_dir, self.counterfactual_results_file), 'a') as f:
                for example_obj in batch_results:
                    f.write(json.dumps(example_obj) + "\n")
                    
    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.task == "concepts":
            self._get_concepts_and_values()
        elif self.task == "counterfactuals":
            self._generate_counterfactuals()
        else:
            raise ValueError(f"Task {self.task} not supported.")