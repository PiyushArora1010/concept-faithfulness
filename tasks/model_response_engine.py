import os
import random
import json
import copy

from tqdm import tqdm
from module.utils import PromptingStrategy
from tasks.engine import Engine

class ModelResponseEngine(Engine):
    def __init__(self, args):
        super().__init__(args)
        self.prompting_strategy = PromptingStrategy(args.cot, args.few_shot, args.knn_rank, args.few_shot_prompt_name, args.add_instr)
        self.output_dir = os.path.join(
            "results", "model_responses", self.dataset_tag, args.output_dir
        )
        self._get_model()

    def _get_original_responses_batch(self, example_indices):
        prompts = []
        for cnt, example_idx in enumerate(example_indices):            
            basic_prompt = self.dataset.format_prompt_basic(example_idx)
            qa_prompt = self.dataset.format_prompt_qa(basic_prompt, self.prompting_strategy, idx=example_idx)
            prompts.extend([qa_prompt] * self.n_completions)
            
        if len(prompts) == 0:
            return
        
        responses = self.model.batch_generate_response(prompts)
        
        for cnt, example_idx in enumerate(example_indices):
            output_dir = os.path.join(
                self.output_dir,
                f"example_{example_idx}",
                f"original"
            )

            for completion_idx in range(self.n_completions):
                global_cnt = cnt * self.n_completions + completion_idx
                response = responses[global_cnt]
                
                file_path = os.path.join(
                    output_dir,
                    f"response_n={completion_idx}.json"
                )
                
                try:
                    answer = self.dataset.extract_answer(
                        response,
                        self.prompting_strategy,
                        idx=example_idx
                    )
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    answer_dict = {
                        "prompt": prompts[global_cnt],
                        "response": response,
                        "answer": answer
                    }
                    with open(file_path, 'w') as f:
                        json.dump(answer_dict, f, indent=4)
                    print(f"Example {example_idx}, Completion {completion_idx}: Answer: {answer}")
                except:
                    answer_dict = {
                        "prompt": prompts[global_cnt],
                        "response": response
                    }

                    file_path = os.path.join("errors", *os.path.normpath(file_path).split(os.path.sep)[1:])
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'w') as f:
                        json.dump(answer_dict, f, indent=4)
                    print(f"Failed to extract answer for example {example_idx}, completion {completion_idx}. Saving empty answer.")

    def _get_original_responses(self):
        batch_size = self.example_batch_size
        
        batch_counter = 0
        example_indices_batch = []
        
        for idx, example_idx in enumerate(self.example_indices):

            if batch_counter >= batch_size:
                self._get_original_responses_batch(example_indices_batch)
                example_indices_batch = []
                batch_counter = 0
                
            if os.path.exists(os.path.join(
                self.output_dir,
                f"example_{example_idx}",
                "original"
            )):
                print(f"Original responses already generated for example {example_idx}. Skipping...")
                continue
                
            example_indices_batch.append(example_idx)
            batch_counter += 1
            
        if len(example_indices_batch) > 0:
            self._get_original_responses_batch(example_indices_batch)

    def _get_counterfactual_responses_batch(self, example_indices, example_intervention_files, example_parsed_counterfactuals):
        prompts = []
        cumulative_counts = []
        for cnt, example_idx in enumerate(example_indices):
            cumulative_counts.append(len(prompts))
            for intervention_file, parsed_counterfactual in zip(example_intervention_files[cnt], example_parsed_counterfactuals[cnt]):
                intrv_str = intervention_file.split(".")[0].split("_")[1]
                intervention_prompt = self.dataset.format_prompt_qa_counterfactual(parsed_counterfactual, self.prompting_strategy, idx=example_idx)
                prompts.extend([intervention_prompt] * self.n_completions)
            
        responses = self.model.batch_generate_response(prompts)
        
        for cnt, example_idx in enumerate(example_indices):
            counterfactual_dir = os.path.join(
                self.output_dir,
                f"example_{example_idx}",
                f"counterfactual"
            )

            for intervention_index in range(len(example_intervention_files[cnt])):
                for completion_idx in range(self.n_completions):
                    global_cnt = cumulative_counts[cnt] + (intervention_index * self.n_completions) + completion_idx
                    
                    response = responses[global_cnt]
                    
                    intrv_str = example_intervention_files[cnt][intervention_index].split(".")[0].split("_")[1]
                    file_path = os.path.join(
                        counterfactual_dir,
                        f"response_counterfactual={intrv_str}_n={completion_idx}.json"
                    )
                    
                    try:
                        answer = self.dataset.extract_answer(
                            response,
                            self.prompting_strategy,
                            idx=example_idx
                        )
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        answer_dict = {
                            "prompt": prompts[global_cnt],
                            "response": response,
                            "answer": answer
                        }
                        with open(file_path, 'w') as f:
                            json.dump(answer_dict, f, indent=4)
                        print(f"Example {example_idx}, Intervention {intrv_str}, Completion {completion_idx}: Answer: {answer}")
            
                    except:
                        answer_dict = {
                            "prompt": prompts[global_cnt],
                            "response": response,
                        }

                        file_path = os.path.join("errors", *os.path.normpath(file_path).split(os.path.sep)[1:])
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        with open(file_path, 'w') as f:
                            json.dump(answer_dict, f, indent=4)
                        print(f"Failed to extract answer for example {example_idx}, intervention {intrv_str}, completion {completion_idx}. Saving empty answer.")

    def _get_counterfactual_responses(self):
        batch_size = self.example_batch_size
        
        batch_counter = 0
        example_indices_batch = []
        example_intervention_files_batch = []
        example_parsed_counterfactuals_batch = []

        for idx, example_idx in enumerate(self.example_indices):
            if batch_counter >= batch_size:
                self._get_counterfactual_responses_batch(
                    example_indices_batch,
                    example_intervention_files_batch,
                    example_parsed_counterfactuals_batch
                )
                example_indices_batch = []
                example_intervention_files_batch = []
                example_parsed_counterfactuals_batch = []
                batch_counter = 0
            
            if os.path.exists(os.path.join(
                self.output_dir,
                f"example_{example_idx}",
                "counterfactual"
            )):
                print(f"Counterfactual responses already generated for example {example_idx}. Skipping...")
                continue
            
            intervention_file_dir = os.path.join(
                self.intervention_dir,
                f"example_{example_idx}"
            )
            
            if not os.path.exists(intervention_file_dir):
                print(f"No intervention files found for example {example_idx}. Skipping...")
                continue
            
            all_intervention_files = [f for f in sorted(os.listdir(intervention_file_dir)) if f.startswith("counterfactual_")]
            parsed_counterfactuals = []
            parsed_intervention_files = []
            
            for intervention_file in all_intervention_files:
                with open(os.path.join(intervention_file_dir, intervention_file), 'r') as f:
                    counterfactual_data = json.load(f)
                parsed_counterfactual = counterfactual_data["parsed_counterfactual"]
                parsed_counterfactuals.append(parsed_counterfactual)
                parsed_intervention_files.append(intervention_file)
                
            if len(parsed_counterfactuals) == 0:
                print(f"No valid counterfactuals found for example {example_idx}. Skipping...")
                continue
                
            example_indices_batch.append(example_idx)
            example_intervention_files_batch.append(parsed_intervention_files)
            example_parsed_counterfactuals_batch.append(parsed_counterfactuals)
            batch_counter += 1
            
        if len(example_indices_batch) > 0:
            self._get_counterfactual_responses_batch(
                example_indices_batch,
                example_intervention_files_batch,
                example_parsed_counterfactuals_batch
            )

    def run(self, task):
        if task == 'original':
            self._get_original_responses()
        elif task == 'counterfactual':
            self._get_counterfactual_responses()
        else:
            raise ValueError(f"Unsupported task: {task}")
        