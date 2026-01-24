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
            "results", "concept_outputs", self.dataset_tag, args.output_dir
        )
        self._get_model()

    def _verify_counterfactual_questions_batch(self, example_indices, example_intervention_files, example_interventions_data):
        prompts = []
        cumulative_counts = []
        # breakpoint()
        for cnt, example_idx in enumerate(example_indices):
            cumulative_counts.append(len(prompts))
            for intervention_file, intervention_data in zip(example_intervention_files[cnt], example_interventions_data[cnt]):
                verification_prompt = self.dataset.format_prompt_qa_verification(
                    intervention_data,
                    verification_base_prompt_name=self.verification_base_prompt_name,
                    idx=example_idx
                )
                prompts.append(verification_prompt)
                # breakpoint()

        responses = self.model.batch_generate_response(prompts)
        
        for cnt, example_idx in enumerate(example_indices):
            counterfactual_dir = os.path.join(
                self.output_dir,
                f"example_{example_idx}",
            )
            example_intervention_data = example_interventions_data[cnt]
            for intervention_index in range(len(example_intervention_files[cnt])):
                global_cnt = cumulative_counts[cnt] + intervention_index
                
                response = responses[global_cnt]
                
                intrv_str = example_intervention_data[intervention_index]["intervention_str"]
                file_path_verification = os.path.join(
                    counterfactual_dir,
                    f"verification_counterfactual_{self.model_tag}_{intrv_str}.json"
                )
                
                try:
                    answer = parse_llm_response_verification(
                        response
                    )
                except:
                    answer = "N/A"
                    
                os.makedirs(os.path.dirname(file_path_verification), exist_ok=True)
                
                answer_dict = {
                    "prompt": prompts[global_cnt],
                    "response": response,
                    "verification": answer
                }
                
                with open(file_path_verification, 'w') as f:
                    json.dump(answer_dict, f, indent=4)
                
                if answer != "N/A":
                    print(f"Example {example_idx}, Intervention {intrv_str}: Successfully verified as {answer}.")
                else:
                    print(f"Example {example_idx}, Intervention {intrv_str}: Verification failed.")
    
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
    
    def run(self, task):
        if task == "verify_counterfactuals":
            self._verify_counterfactual_questions()
        else:
            raise NotImplementedError(f"Task {task} not implemented in VerificationEngine.")