import os
import json

from module.utils import parse_llm_response_implied_concepts
from tasks.engine import Engine

class ImpliedConceptsEngine(Engine):
    def __init__(self, args):
        super().__init__(args)
        self.output_dir = os.path.join(
            "results", "implied_concepts", self.dataset_tag, args.output_dir
        )
        self._get_model()
            
    def _identify_implied_concepts_batch(self, sub_dir, example_indices, example_concepts, example_concept_values, example_interventions, example_responses):
        prompts = []
        concepts_to_check_len = []
        cumulative_counts = []
        response_cumulative_counts = []

        for cnt, example_idx in enumerate(example_indices):
            concepts = example_concepts[cnt]
            concept_values = example_concept_values[cnt]
            response_dict = example_responses[cnt]
            intervention_dict = example_interventions[cnt]

            cumulative_counts.append(len(prompts))
            response_cumulative_counts.append(len(concepts_to_check_len))

            for response_id in sorted(response_dict.keys()):
                example_dir = os.path.join(
                    self.output_dir,
                    f"example_{example_idx}",
                    sub_dir
                )
                if not os.path.exists(example_dir):
                    os.makedirs(example_dir)

                concepts_to_check = concepts
                values_concepts_to_check = concept_values

                if "original" in sub_dir:
                    basic_prompt = self.dataset.format_prompt_basic(
                        example_idx,
                        double_space=False
                    )
                else:
                    intervention = response_id.split("_")[1].split("=")[1]
                    counterfactual_file = os.path.join(
                        self.intervention_dir,
                        f"example_{example_idx}",
                        f"counterfactual_{intervention}.json"
                    )
                    counterfactual_dict = intervention_dict[counterfactual_file]

                    basic_prompt = self.dataset.format_question_counterfactual(
                        counterfactual_dict["parsed_counterfactual"],
                        double_space=False
                    )

                    if_not_known = [x != "UNKNOWN" for x in counterfactual_dict["new_values"]]
                    concepts_to_check = [f for f, i in zip(concepts, if_not_known) if i]
                    values_concepts_to_check = [f for f, i in zip(concept_values, if_not_known) if i]

                model_response = response_dict[response_id]["response"]
                model_answer = response_dict[response_id]["answer"]

                prompt = self.dataset.format_prompt_implied_concepts(
                    self.implied_concepts_base_prompt_name,
                    concepts_to_check,
                    values_concepts_to_check,
                    basic_prompt,
                    model_response,
                    model_answer
                )

                concepts_to_check_len.append(len(concepts_to_check))
                prompts.extend([prompt] * self.n_completions)

        all_responses = self.model.batch_generate_response(prompts)

        for cnt, example_idx in enumerate(example_indices):
            response_dict = example_responses[cnt]

            for response_idx, response_id in enumerate(sorted(response_dict.keys())):
                global_cnt = cumulative_counts[cnt] + (response_idx * self.n_completions)

                response_list = []
                concept_decision_list = []

                for r in range(self.n_completions):
                    response = all_responses[global_cnt + r]
                    try:
                        concept_decision, parsed_response = parse_llm_response_implied_concepts(
                            response,
                            concepts_to_check_len[
                                response_cumulative_counts[cnt] + response_idx
                            ]
                        )
                        response_list.append(parsed_response)
                        concept_decision_list.append(concept_decision)
                    except Exception as e:
                        print(
                            f"Failed to parse implied concepts response for example "
                            f"{example_idx}, response {response_id}, completion {r}. "
                            f"Saving empty response."
                        )
                        response_list.append(response)
                        concept_decision_list.append({"error": str(e)})

                output_path = os.path.join(
                    self.output_dir,
                    f"example_{example_idx}",
                    sub_dir,
                    f"implied_concepts_{response_id}"
                )

                fd_dict = {
                    "prompt": prompts[global_cnt],
                    "concept_decisions": concept_decision_list,
                    "responses": response_list
                }

                with open(output_path, "w") as f:
                    json.dump(fd_dict, f, indent=4)
                    
                print(f"Saved implied concepts for example {example_idx}, response {response_id}.")
        
    def _identify_implied_concepts(self, sub_dir):
        batch_size = self.example_batch_size
        
        batch_counter = 0
        example_indices_batch = []
        example_concepts_batch = []
        example_concept_values_batch = []
        example_interventions_batch = []
        example_responses_batch = []
        
        for idx, example_idx in enumerate(self.example_indices):
            if batch_counter >= batch_size:
                self._identify_implied_concepts_batch(
                    sub_dir,
                    example_indices_batch,
                    example_concepts_batch,
                    example_concept_values_batch,
                    example_interventions_batch,
                    example_responses_batch
                )
                example_indices_batch = []
                example_concepts_batch = []
                example_concept_values_batch = []
                example_interventions_batch = []
                example_responses_batch = []
                batch_counter = 0

            if os.path.exists(os.path.join(
                self.output_dir,
                f"example_{example_idx}",
                sub_dir
            )):
                print(f"Implied concepts already identified for example {example_idx} in {sub_dir}. Skipping...")
                continue

            if not os.path.exists(os.path.join(
                self.response_dir,
                f"example_{example_idx}",
                sub_dir
            )):
                print(f"No responses found for example {example_idx} in {sub_dir}. Skipping...")
                continue

            concept_path = os.path.join(
                self.intervention_dir,
                f"example_{example_idx}",
                "concepts.json"
            )
            if not os.path.exists(concept_path):
                print(f"No concepts found for example {example_idx}. Skipping...")
                continue

            with open(os.path.join(
                self.intervention_dir,
                f"example_{example_idx}",
                "concepts.json"
            ), 'r') as f:
                concepts = json.load(f)
            
            concept_settings_path = os.path.join(
                self.intervention_dir,
                f"example_{example_idx}",
                "concept_settings.json"
            )
            if not os.path.exists(concept_settings_path):
                print(f"No concept settings found for example {example_idx}. Skipping...")
                continue
            
            with open(os.path.join(
                self.intervention_dir,
                f"example_{example_idx}",
                "concept_settings.json"
            ), 'r') as f:
                concept_values = json.load(f)
            
            # LOAD INTERVENTIONS
            intervention_dir = os.path.join(
                self.intervention_dir,
                f"example_{example_idx}"
            )
            all_intervention_files = [f for f in sorted(os.listdir(intervention_dir)) if f.startswith("counterfactual_")]
            intervention_dict = {}
            for intervention_file in all_intervention_files:
                with open(os.path.join(intervention_dir, intervention_file), 'r') as f:
                    counterfactual_data = json.load(f)
                intervention_dict[os.path.join(intervention_dir, intervention_file)] = counterfactual_data
            if len(intervention_dict) == 0:
                print(f"No interventions found for example {example_idx}. Skipping...")
                continue
            
            response_dir = os.path.join(
                self.response_dir,
                f"example_{example_idx}",
                sub_dir
            )
            response_files = [f for f in sorted(os.listdir(response_dir)) if f.startswith("response_")]
            response_dict = {}
            for response_file in response_files:
                with open(os.path.join(response_dir, response_file), 'r') as f:
                    response_data = json.load(f)
                response_dict[response_file] = response_data
            
            if len(response_dict) == 0:
                print(f"No responses found for example {example_idx} in {sub_dir}. Skipping...")
                continue
            
            example_indices_batch.append(example_idx)
            example_concepts_batch.append(concepts)
            example_concept_values_batch.append(concept_values)
            example_interventions_batch.append(intervention_dict)
            example_responses_batch.append(response_dict)
            batch_counter += 1
            
        if len(example_indices_batch) > 0:
            self._identify_implied_concepts_batch(
                sub_dir,
                example_indices_batch,
                example_concepts_batch,
                example_concept_values_batch,
                example_interventions_batch,
                example_responses_batch
            )
    
    def run(self, task):
        if task == 'original':
            self._identify_implied_concepts('original')
        elif task == 'counterfactual':
            self._identify_implied_concepts('counterfactual')
        