import os
import glob
import json
import copy

from module.utils import parse_llm_response_verification

from tasks.engine import Engine

class VerificationEngine(Engine):
    def __init__(self, args):
        super().__init__(args)
        self.output_dir = os.path.join(
            "verified_results", "concept_outputs", self.dataset_tag, args.output_dir
        )
        self._get_model()
        
    def _verify_counterfactual_questions_batch(self, example_indices, example_intervention_files, example_interventions_data):
        prompts = []
        cumulative_counts = []

        for cnt, example_idx in enumerate(example_indices):
            cumulative_counts.append(len(prompts))
            for intervention_file, intervention_data in zip(example_intervention_files[cnt], example_interventions_data[cnt]):
                verification_prompt_question = self.dataset.format_prompt_question_verification(
                    intervention_data,
                    verification_base_prompt_name=self.verification_question_base_prompt_name,
                    idx=example_idx
                )
                verification_prompt_anschoices = self.dataset.format_prompt_anschoices_verification(
                    intervention_data,
                    verification_base_prompt_name=self.verification_anschoices_base_prompt_name,
                    idx=example_idx
                )
                prompts.append(verification_prompt_question)
                prompts.append(verification_prompt_anschoices)
                # breakpoint()

        responses = self.model.batch_generate_response(prompts)
        
        for cnt, example_idx in enumerate(example_indices):
            counterfactual_dir = os.path.join(
                self.output_dir,
                f"example_{example_idx}",
            )
            example_intervention_data = example_interventions_data[cnt]
            for intervention_index in range(len(example_intervention_files[cnt])):
                # Each intervention has 2 prompts (question and answer choices)
                global_cnt_question = cumulative_counts[cnt] + (intervention_index * 2)
                global_cnt_anschoices = cumulative_counts[cnt] + (intervention_index * 2) + 1
                
                response_question = responses[global_cnt_question]
                response_anschoices = responses[global_cnt_anschoices]
                
                intrv_str = example_intervention_data[intervention_index]["intervention_str"]
                file_path_verification = os.path.join(
                    counterfactual_dir,
                    f"verification_counterfactual_{self.model_tag}_{intrv_str}.json"
                )
                
                # Parse both verification responses
                try:
                    answer_question = parse_llm_response_verification(response_question)
                except:
                    answer_question = "N/A"
                
                try:
                    answer_anschoices = parse_llm_response_verification(response_anschoices)
                except:
                    answer_anschoices = "N/A"
                
                # Determine overall verification status
                # Both must be "YES" for overall verification to be "YES"
                if answer_question == "YES" and answer_anschoices == "YES":
                    overall_verification = "YES"
                elif answer_question == "NO" or answer_anschoices == "NO":
                    overall_verification = "NO"
                else:
                    overall_verification = "N/A"
                
                os.makedirs(os.path.dirname(file_path_verification), exist_ok=True)
                
                answer_dict = {
                    "question_verification": {
                        "prompt": prompts[global_cnt_question],
                        "response": response_question,
                        "verification": answer_question
                    },
                    "anschoices_verification": {
                        "prompt": prompts[global_cnt_anschoices],
                        "response": response_anschoices,
                        "verification": answer_anschoices
                    },
                    "verification": overall_verification
                }
                
                with open(file_path_verification, 'w') as f:
                    json.dump(answer_dict, f, indent=4)
                
                if overall_verification != "N/A":
                    print(f"Example {example_idx}, Intervention {intrv_str}: Successfully verified as {overall_verification} (Question: {answer_question}, AnsChoices: {answer_anschoices}).")
                else:
                    print(f"Example {example_idx}, Intervention {intrv_str}: Verification failed (Question: {answer_question}, AnsChoices: {answer_anschoices}).")
                
    def _verify_counterfactual_questions(self):
        batch_size = self.example_batch_size
        
        batch_counter = 0
        example_indices_batch = []
        example_intervention_files_batch = []
        example_interventions_data_batch = []

        for idx, example_idx in enumerate(self.example_indices):
            if batch_counter >= batch_size:
                self._verify_counterfactual_questions_batch(
                    example_indices_batch,
                    example_intervention_files_batch,
                    example_interventions_data_batch
                )
                example_indices_batch = []
                example_intervention_files_batch = []
                example_interventions_data_batch = []
                batch_counter = 0
            
            intervention_file_dir = os.path.join(
                self.output_dir,
                f"example_{example_idx}"
            )
            
            if not os.path.exists(intervention_file_dir):
                print(f"No intervention files found for example {example_idx}. Skipping...")
                continue
            
            all_intervention_files = glob.glob(os.path.join(intervention_file_dir, "counterfactual_*1*.json"))
            
            if len(all_intervention_files) == 0:
                print(f"No intervention files found for example {example_idx}. Skipping...")
                continue

            already_verified_files = glob.glob(os.path.join(
                intervention_file_dir,
                f"verification_counterfactual_{self.model_tag}_*.json"
            ))
            
            # Filter out already verified files
            all_intervention_files = [f for f in all_intervention_files if f.replace("counterfactual_", "verification_counterfactual_") + f"_{self.model_tag}.json" not in already_verified_files]
            
            if len(all_intervention_files) == 0:
                print(f"All intervention files for example {example_idx} have already been verified. Skipping...")
                continue
            
            parsed_counterfactuals = []
            parsed_intervention_files = []
            
            for intervention_file in all_intervention_files:
                with open(intervention_file, 'r') as f:
                    counterfactual_data = json.load(f)

                parsed_counterfactuals.append(counterfactual_data)
                parsed_intervention_files.append(intervention_file)
                
            example_indices_batch.append(example_idx)
            example_intervention_files_batch.append(parsed_intervention_files)
            example_interventions_data_batch.append(parsed_counterfactuals)
            batch_counter += 1
            
        if len(example_indices_batch) > 0:
            self._verify_counterfactual_questions_batch(
                example_indices_batch,
                example_intervention_files_batch,
                example_interventions_data_batch
            )
    
    def _verify_concepts_batch(self, example_indices, example_concepts):
        prompts = []
        cumulative_counts = []
        
        for cnt, example_idx in enumerate(example_indices):
            cumulative_counts.append(len(prompts))
            
            for concept in example_concepts[cnt]:
                verification_prompt = self.dataset.format_prompt_concept_verification(
                    concept,
                    verification_base_prompt_name=self.verification_base_prompt_name,
                    idx=example_idx
                )
                # breakpoint()
                prompts.append(verification_prompt)
                
        responses = self.model.batch_generate_response(prompts)
        
        for cnt, example_idx in enumerate(example_indices):
            example_dir = os.path.join(
                self.output_dir,
                f"example_{example_idx}",
            )
            example_save_path = os.path.join(
                example_dir,
                f"verification_concepts_{self.model_tag}.json"
            )
            example_concept_results = []
            print(self.dataset.format_question_info(example_idx, True, 0))
            for concept_index, concept in enumerate(example_concepts[cnt]):
                global_cnt = cumulative_counts[cnt] + concept_index
                
                response = responses[global_cnt]
                
                try:
                    concept_decision = parse_llm_response_verification(
                        response
                    )
                except:
                    concept_decision = "N/A"
                
                print(f"Example {example_idx}, Concept {concept}: Verification result - {concept_decision}")
                concept_result = {
                    "concept": concept,
                    "response": response,
                    "verification": concept_decision
                }
                example_concept_results.append(concept_result)
            
            
            
            os.makedirs(os.path.dirname(example_save_path), exist_ok=True)
            with open(example_save_path, 'w') as f:
                json.dump(example_concept_results, f, indent=4)

    def _verify_concepts(self):
        batch_size = self.example_batch_size
        
        batch_counter = 0
        example_indices_batch = []
        example_concepts_batch = []

        for idx, example_idx in enumerate(self.example_indices):
            if batch_counter >= batch_size:
                self._verify_concepts_batch(
                    example_indices_batch,
                    example_concepts_batch
                )
                example_indices_batch = []
                example_concepts_batch = []
                batch_counter = 0
            
            concept_file_path = os.path.join(
                self.output_dir,
                f"example_{example_idx}",
                "concept_settings.json"
            )
            
            verification_file_path = os.path.join(
                self.output_dir,
                f"example_{example_idx}",
                f"verification_concepts_{self.model_tag}.json"
            )
            
            if os.path.exists(verification_file_path):
                print(f"Concepts for example {example_idx} have already been verified. Skipping...")
                continue
            
            if not os.path.exists(concept_file_path):
                print(f"No concept file found for example {example_idx}. Skipping...")
                continue
            
            with open(concept_file_path, 'r') as f:
                concept_data = json.load(f)
                
            concepts = [[entry["current_setting"], entry["new_settings"][0]] for entry in concept_data]
            
            example_indices_batch.append(example_idx)
            example_concepts_batch.append(concepts)
            batch_counter += 1
            
        if len(example_indices_batch) > 0:
            self._verify_concepts_batch(
                example_indices_batch,
                example_concepts_batch
            )

    def run(self, task):
        if task == "verify_interventions":
            self._verify_counterfactual_questions()
        elif task == "verify_concepts":
            self._verify_concepts()
        else:
            raise NotImplementedError(f"Task {task} not implemented in VerificationEngine.")