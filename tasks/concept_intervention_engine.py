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
        # self._get_model()
        
        self.concept_results_file = "concept_results.json"
        with open(os.path.join("prompts", self.dataset_tag, f"{self.concept_id_base_prompt_name}.txt"), 'r') as f:
            self.concept_id_base_prompt = f.read()
            
        with open(os.path.join("prompts", self.dataset_tag, f"{self.concept_values_base_prompt_name}.txt"), 'r') as f:
            self.concept_values_base_prompt = f.read()
        
        self.counterfactual_results_file = "counterfactual_results.json"
        with open(os.path.join("prompts", self.dataset_tag, f"{self.counterfactual_gen_base_prompt_name}.txt"), 'r') as f:
            self.counterfactual_gen_base_prompt = f.read()

    def parse_concepts(self, response):
        """
        Extract lines that start with an integer followed by a dot.
        Example: '1. The identity of the individual'
        """
        pattern = r'^\s*\d+\.\s*(.+)$'
        
        concepts = []
        for line in response.split("\n"):
            match = re.match(pattern, line)
            if match:
                concepts.append(match.group(1).strip())
        
        return concepts
            
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

    def _get_concept_ids_batch(self, example_indices):
        prompts = []
        for cnt, example_idx in enumerate(example_indices):
            prompt = self.concept_id_base_prompt.strip() + "\n" + self.dataset[example_idx]["question"]
            prompts.append(prompt)
        
        if len(prompts) == 0:
            return
        
        # breakpoint()
        responses = self.model.batch_generate_response(prompts)
        
        batched_concept_ids = {}
        for cnt, example_idx in enumerate(example_indices):
            response = responses[cnt]
            print(f"LLM Response for example {example_idx}:\n{response}\n")
            try:
                concepts = self.parse_concepts(response)
                batched_concept_ids[example_idx] = concepts
            except:
                print(f"Error parsing concepts for example {example_idx}. Saving response for analysis.")
                concepts = None
                
        return batched_concept_ids

    def _get_concept_values_batch(self, example_indices, batched_concept_ids):
        prompts = []
        for cnt, example_idx in enumerate(example_indices):
            prompt = self.concept_values_base_prompt + "\n" + self.dataset[example_idx]["question"] + "\n"
            concept_list = "Concept List:\n"
            for i, concept in enumerate(batched_concept_ids[example_idx]):
                concept_list += f"{i+1}. {concept}\n"
            prompt += concept_list[:-1]
            
            prompts.append(prompt)
            
        if len(prompts) == 0:
            return
        # breakpoint()
        responses = self.model.batch_generate_response(prompts)
        
        batched_concept_values = {}
        for cnt, example_idx in enumerate(example_indices):
            response = responses[cnt]
            print(f"LLM Response for concept settings for example {example_idx}:\n{response}\n")
            
            try:
                original_values, alternaltive_values = self.parse_concept_values(response)
                
                assert len(original_values) == len(alternaltive_values), "Number of original values does not match number of alternative value sets."
                assert len(original_values) == len(batched_concept_ids[example_idx]), "Number of concept settings does not match number of concepts."
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

                # concept_ids = self._get_concept_ids_batch(example_indices_batch)
                concept_ids = {example_idx: ["Concept 1", "Concept 2"] for example_idx in example_indices_batch} # Dummy concept IDs for testing
                concept_values = self._get_concept_values_batch([i for i in example_indices_batch], concept_ids)
                
                with open(os.path.join(self.output_dir, self.concept_results_file), 'a') as f:
                    for idx in example_indices_batch:
                        result = {
                            "example_idx": idx,
                            "concept_ids": concept_ids[idx],
                            "concept_values": concept_values[idx]
                        }
                        f.write(json.dumps(result) + "\n")
                
                example_indices_batch = []
                batch_counter = 0
            
            example_indices_batch.append(example_idx)
            batch_counter += 1
            
        if len(example_indices_batch) > 0:
            concept_ids = self._get_concept_ids_batch(example_indices_batch)
            concept_values = self._get_concept_values_batch([i for i in example_indices_batch], concept_ids)
            
            with open(os.path.join(self.output_dir, self.concept_results_file), 'a') as f:
                for idx in example_indices_batch:
                    result = {
                        "example_idx": idx,
                        "concept_ids": concept_ids[idx],
                        "concept_values": concept_values[idx]
                    }
                    f.write(json.dumps(result) + "\n")
                    
    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.task == "concepts":
            self._get_concepts_and_values()
        else:
            raise ValueError(f"Task {self.task} not supported.")