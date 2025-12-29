import os
import json
import copy

from module.utils import parse_llm_response_concepts_and_categories, parse_llm_response_factor_settings, enumerate_interventions

from tasks.engine import Engine

class ConceptInterventionEngine(Engine):
    def __init__(self, args):
        super().__init__(args)
        
    def _get_concept_ids_batch(self, example_indices):
        prompts = []
        for cnt, example_idx in enumerate(example_indices):
            prompt = self.dataset.format_prompt_concept_id(
                example_idx,
                self.concept_id_base_prompt_name
            )
            prompts.append(prompt)
        
        if len(prompts) == 0:
            return
        
        responses = self.model.batch_generate_response(prompts)
        for cnt, example_idx in enumerate(example_indices):

            example_dir = os.path.join(self.output_dir, f"example_{example_idx}")
            if not os.path.exists(example_dir):
                os.makedirs(example_dir)
            
            response = responses[cnt]
            print(f"LLM Response for example {example_idx}:\n{response}\n")
                
            concept_path = os.path.join(example_dir, "concepts.json")
            category_path = os.path.join(example_dir, "categories.json")
            
            concepts, categories = parse_llm_response_concepts_and_categories(response)
            with open(concept_path, 'w') as f:
                json.dump(concepts, f, indent=4)
            with open(category_path, 'w') as f:
                json.dump(categories, f, indent=4)
                
            print(f"Saved concepts and categories for example {example_idx} to {concept_path} and {category_path}.")
    
    def _get_concept_ids(self):
        batch_size = self.batch_size
        batch_counter = 0
        example_indices_batch = []
        for idx, example_idx in enumerate(range(self.example_indices[0], self.example_indices[-1] + 1)):
            if batch_counter >= batch_size:
                self._get_concept_ids_batch(example_indices_batch)
                example_indices_batch = []
                batch_counter = 0
            
            example_dir = os.path.join(self.output_dir, f"example_{example_idx}")
            if os.path.exists(os.path.join(example_dir, "concepts.json")):
                print(f"Concepts already generated for example {example_idx}. Skipping...")
                continue
            
            example_indices_batch.append(example_idx)
            batch_counter += 1
        if len(example_indices_batch) > 0:
            self._get_concept_ids_batch(example_indices_batch)
    
    def _get_intervention_sets_batch(self, example_indices):
        prompts = []
        for cnt, example_idx in enumerate(example_indices):
            example_dir = os.path.join(self.output_dir, f"example_{example_idx}")
            concept_path = os.path.join(example_dir, "concepts.json")
            with open(concept_path, 'r') as f:
                concepts = json.load(f)

            prompt = self.dataset.format_prompt_concept_values(
                example_idx,
                self.concept_values_base_prompt_name,
                concepts
            )
            prompts.append(prompt)
            
        if len(prompts) == 0:
            return
        
        responses = self.model.batch_generate_response(prompts)
        
        for cnt, example_idx in enumerate(example_indices):
            example_dir = os.path.join(self.output_dir, f"example_{example_idx}")
            if not os.path.exists(example_dir):
                os.makedirs(example_dir)

            response = responses[cnt]
            print(f"LLM Response for concept settings for example {example_idx}:\n{response}\n")
            
            concept_settings_path = os.path.join(example_dir, "concept_settings.json")
            concept_settings = parse_llm_response_factor_settings(response)
            with open(concept_settings_path, 'w') as f:
                json.dump(concept_settings, f, indent=4)
            print(f"Saved concept settings for example {example_idx} to {concept_settings_path}.")

    def _get_intervention_sets(self):
        batch_counter = 0   
        example_indices_batch = []
        for idx, example_idx in enumerate(range(self.example_indices[0], self.example_indices[-1] + 1)):
            if batch_counter >= self.batch_size:
                self._get_intervention_sets_batch(example_indices_batch)
                example_indices_batch = []
                batch_counter = 0
            
            example_dir = os.path.join(self.output_dir, f"example_{example_idx}")
            if os.path.exists(os.path.join(example_dir, "concept_settings.json")):
                print(f"Concept settings already generated for example {example_idx}. Skipping...")
                continue
            
            example_indices_batch.append(example_idx)
            batch_counter += 1
        if len(example_indices_batch) > 0:
            self._get_intervention_sets_batch(example_indices_batch)


    def _apply_interventions_batch(self, example_indices, concepts_list, concept_settings_list):
        counterfactual_gen_dics = []
        for cnt, example_idx in enumerate(example_indices):
            example_dir = os.path.join(self.output_dir, f"example_{example_idx}")
            existing_interventions = [x.split('.')[0].split('_')[1] for x in os.listdir(example_dir) if x.startswith('counterfactual_')]
            if self.only_concept_removals:
                for factor_setting in concept_settings_list[cnt]:
                    factor_setting["new_settings"] = ["UNKNOWN"]
            
            if self.include_unknown_concept_values and not self.only_concept_removals:
                for factor_setting in concept_settings_list[cnt]:
                    if "UNKNOWN" not in factor_setting["new_settings"]:
                        factor_setting["new_settings"].append("UNKNOWN")
                        
            intervention_list = enumerate_interventions(concepts_list[cnt], concept_settings_list[cnt], k_hop=1, include_no_intervention=False, mark_removals=True)
            intervention_list = [x for x in intervention_list if x not in existing_interventions]

            if len(intervention_list) == 0:
                print(f"All interventions already generated for example {example_idx}. Skipping...")
                continue
            
            counterfactual_gen_dic = [self._apply_single_intervention(example_idx, intervention_str, concepts_list[cnt], concept_settings_list[cnt]) for intervention_str in intervention_list]
            counterfactual_gen_dics.extend(counterfactual_gen_dic)
            
        if len(counterfactual_gen_dics) == 0:
            return
        
        counterfactual_responses_examples = self.model.batch_generate_response(
            [x["counterfactual_gen_prompt"] for x in counterfactual_gen_dics]
        )
        
        for idx, counterfactual_gen_dic in enumerate(counterfactual_gen_dics):
            example_idx = counterfactual_gen_dic["example_idx"]
            del counterfactual_gen_dic["example_idx"]
            
            s = counterfactual_responses_examples[idx]
            s = s[s.index("Edited Context"):]
            counterfactual_gen_dic["counterfactual"] = s
            
            try:
                counterfactual_gen_dic["parsed_counterfactual"] = self.dataset.parse_counterfactual_output(
                    counterfactual_gen_dic["counterfactual"]
                )
            except Exception as e:
                counterfactual_gen_dic["parsed_counterfactual"] = e
                print(f"Error parsing counterfactual for example {example_idx}, intervention {counterfactual_gen_dic['intervention_str']}: {e}")
                
            output_path = counterfactual_gen_dic["output_path"]
            del counterfactual_gen_dic["output_path"]

            with open(output_path, 'w') as f:
                json.dump(counterfactual_gen_dic, f, indent=4)
            print(f"Saved counterfactual for example {example_idx}, intervention {counterfactual_gen_dic['intervention_str']} to {output_path}.")
            
    def _apply_single_intervention(self, example_idx, intervention_str, concepts, concept_settings):
        old_values = [x["current_setting"] for x in concept_settings]
        new_values = copy.deepcopy(old_values)
        example_dir = os.path.join(self.output_dir, f"example_{example_idx}")
        for idx, val in enumerate(intervention_str):
            if val == '-':
                new_values[idx] = "UNKNOWN"
            else:
                val_int = int(val)
                if val_int:
                    new_values[idx] = concept_settings[idx]["new_settings"][val_int - 1]

        intervention_bool = [True if intervention_str[i] != "0" else False for i in range(len(intervention_str))]
        counterfactual_gen_prompt = self.dataset.format_prompt_counterfactual_gen(
            example_idx, 
            self.counterfactual_gen_base_prompt_name, 
            concepts, intervention_bool, 
            new_values, 
            old_values
        )
        
        return {
            "example_idx": example_idx,
            "intervention_str": intervention_str,
            "old_values": old_values,
            "new_values": new_values,
            "counterfactual": None,
            "counterfactual_gen_prompt": counterfactual_gen_prompt,
            "parsed_counterfactual": None,
            "output_path": os.path.join(example_dir, f"counterfactual_{intervention_str}.json")
        }

    def _apply_interventions(self):
        batch_size = self.batch_size
        
        batch_counter = 0
        example_indices_batch = []
        concepts_list = []
        concept_settings_list = []
        
        for idx, example_idx in enumerate(range(self.example_indices[0], self.example_indices[-1] + 1)):

            if batch_counter >= self.batch_size:
                self._apply_interventions_batch(example_indices_batch, concepts_list, concept_settings_list)
                batch_counter = 0
                example_indices_batch = []
                concepts_list = []
                concept_settings_list = []
            
            example_dir = os.path.join(self.output_dir, f"example_{example_idx}")
            concept_path = os.path.join(example_dir, "concepts.json")
            concept_settings_path = os.path.join(example_dir, "concept_settings.json")
            
            with open(concept_path, 'r') as f:
                concepts = json.load(f)
            with open(concept_settings_path, 'r') as f:
                concept_settings = json.load(f)
                
            if concept_settings[0]["current_setting"] == "N/A":
                print(f"No valid concept settings for example {example_idx}. Skipping intervention generation...")
                continue
            
            example_indices_batch.append(example_idx)
            concepts_list.append(concepts)
            concept_settings_list.append(concept_settings)
            
            batch_counter += len(concepts)
            
        if batch_counter > 0:
            self._apply_interventions_batch(example_indices_batch, concepts_list, concept_settings_list)
            
    def run(self, task):
        if task == 'get_concept_ids':
            self._get_concept_ids()
        elif task == 'get_intervention_sets':
            self._get_intervention_sets()
        elif task == 'apply_interventions':
            self._apply_interventions()
        else:
            raise ValueError(f"Unsupported task: {task}")
        