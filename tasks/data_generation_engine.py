import os
import json
import copy
import glob

from module.utils import PromptingStrategy

from tasks.engine import Engine

class DataGenerationEngine(Engine):
    def __init__(self, args):
        super().__init__(args)
        self.output_dir = os.path.join(
            "results", "data_generation", self.dataset_tag, args.output_dir
        )
        self._get_model()

    def create_explanation_prompt(self, question, answer, concepts):
        concepts_str = ", ".join(concepts)
        
        prompt = f"""You are given a question, its correct answer, and a specific set of concepts.

Your task is to generate an "explanation" that shows how to arrive at the given answer from the multiple choice question given a list of "concepts" extracted from the question.

IMPORTANT:
- Your explanation must use ALL of the provided concepts
- Your explanation must use ONLY the provided concepts (do not introduce any other concepts)
- The explanation should demonstrate the reasoning path from question to answer using exclusively these concepts
- Do not mention that concepts were provided to you - write as if this is your organic reasoning
- Make sure your explanation clearly demonstrates that these concepts were influential and essential in determining the answer

Input:
- Question: 
{question}
- Answer: {answer}
- Concepts to use: 
{concepts_str}

Output your explanation inside <EXPLANATION></EXPLANATION> tags."""
    
        return prompt

    def _extract_explanation(self, response):
        start_tag = "<EXPLANATION>"
        end_tag = "</EXPLANATION>"
        start_idx = response.find(start_tag)
        end_idx = response.find(end_tag)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            explanation = response[start_idx + len(start_tag) : end_idx].strip()
            return explanation
        else:
            return None

    def _get_explanations_batch(self, example_indices, concepts_list, concept_values_list, answer_list):
        prompts = []
        for cnt, example_idx in enumerate(example_indices):
            concepts = concepts_list[cnt]
            concept_values = concept_values_list[cnt]
            answer = answer_list[cnt]
            
            basic_prompt = self.dataset.format_prompt_basic(example_idx)
            explanation_prompt = self.create_explanation_prompt(
                question=basic_prompt,
                answer=answer,
                concepts=[f"{concepts[i]}: {concept_values[i]['current_setting']}" for i in range(len(concepts))]
            )
            prompts.append(explanation_prompt)
                
        if len(prompts) == 0:
            return
        
        responses = self.model.batch_generate_response(prompts)
        
        for cnt, example_idx in enumerate(example_indices):
            explanation = self._extract_explanation(responses[cnt])
            if explanation is None:
                print(f"Failed to extract explanation for example {example_idx}.")
                continue
            example_dir = os.path.join(
                self.output_dir,
                f"example_{example_idx}"
            )
            
            os.makedirs(example_dir, exist_ok=True)
            explanation_file = os.path.join(
                example_dir,
                f"explanation.json"
            )
            with open(explanation_file, 'w') as f:
                json.dump({
                    "example_idx": example_idx,
                    "prompt": prompts[cnt],
                    "concept_values": concept_values_list[cnt],
                    "question": self.dataset.format_prompt_basic(example_idx),
                    "answer": answer_list[cnt],
                    "explanation": explanation
                }, f, indent=4)
            print(f"Example {example_idx}: Explanation extracted and saved.")

    def _get_explanations(self, task):
        batch_size = self.example_batch_size
        
        batch_counter = 0
        example_indices_batch = []
        concepts_list_batch = []
        concept_values_list_batch = []
        answer_list_batch = []
        
        for idx, example_idx in enumerate(self.example_indices):

            if batch_counter >= batch_size:
                self._get_explanations_batch(
                    example_indices_batch,
                    concepts_list_batch,
                    concept_values_list_batch,
                    answer_list_batch
                )
                example_indices_batch = []
                concepts_list_batch = []
                concept_values_list_batch = []
                answer_list_batch = []
                batch_counter = 0
                
            example_dir = os.path.join(
                self.output_dir,
                f"example_{example_idx}"
            )
            explanation_file = os.path.join(
                example_dir,
                f"explanation.json"
            )
            if os.path.exists(explanation_file):
                print(f"Explanation already generated for example {example_idx}. Skipping...")
                continue
                
            intervention_path = os.path.join(
                self.intervention_dir,
                f"example_{example_idx}",
            )
            
            if not os.path.exists(intervention_path):
                print(f"Intervention data not found for example {example_idx}. Skipping...")
                continue
            
            concepts_file = os.path.join(
                intervention_path,
                "concepts.json"
            )
            if not os.path.exists(concepts_file):
                print(f"Concepts file not found for example {example_idx}. Skipping...")
                continue
            
            concept_settings_file = os.path.join(
                intervention_path,
                "concept_settings.json"
            )
            if not os.path.exists(concept_settings_file):
                print(f"Concept settings file not found for example {example_idx}. Skipping...")
                continue
            
            with open(concepts_file, 'r') as f:
                concepts = json.load(f)
            
            with open(concept_settings_file, 'r') as f:
                concept_values = json.load(f)

            response_dir = os.path.join(
                self.response_dir,
                f"example_{example_idx}"
            )

            original_response_file = os.path.join(
                response_dir,
                "original",
                "response_n=0.json"
            )
            if not os.path.exists(original_response_file):
                print(f"Original response file not found for example {example_idx}. Skipping...")
                continue
            
            with open(original_response_file, 'r') as f:
                answer = json.load(f)["answer"]

            if task == "all_concepts":
                concepts = concepts
                concept_values = concept_values
            elif task == "implied_concepts":
                counterfactual_response_files = glob.glob(
                    os.path.join(response_dir, "counterfactual", "response_counterfactual=*1*_n=0.json")
                )
                if len(counterfactual_response_files) == 0:
                    print(f"No counterfactual response files found for example {example_idx}. Skipping...")
                    continue
                    
                valid_concepts_bool = [False] * len(concepts)
                for cf_file in counterfactual_response_files:
                    with open(cf_file, 'r') as f:
                        cf_answer = json.load(f)["answer"]
                    intenvention_str = cf_file.split("counterfactual=")[1].split("_n=0.json")[0]
                    concept_idx = intenvention_str.find("1")
                    if cf_answer != answer:
                        valid_concepts_bool[concept_idx] = True
                
                concepts = [concepts[i] for i in range(len(concepts)) if valid_concepts_bool[i]]
                concept_values = [concept_values[i] for i in range(len(concept_values)) if valid_concepts_bool[i]]
                
                if len(concepts) == 0:
                    print(f"No influential concepts found for example {example_idx}. Skipping...")
                    continue
            else:
                raise ValueError(f"Unknown task: {task}")

            example_indices_batch.append(example_idx)
            concepts_list_batch.append(concepts)
            concept_values_list_batch.append(concept_values)
            answer_list_batch.append(answer)
            batch_counter += 1
            
        if len(example_indices_batch) > 0:
            self._get_explanations_batch(
                example_indices_batch,
                concepts_list_batch,
                concept_values_list_batch,
                answer_list_batch
            )
            
    def run(self, task):
        if task == "all_concepts" or task == "implied_concepts":
            self._get_explanations(task)
        else:
            raise ValueError(f"Unknown task: {task}")