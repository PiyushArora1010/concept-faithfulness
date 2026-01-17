import os
import sys
import random

from tasks.engine import Engine
from module.datasets.dataset import HF_Dataset
from module.utils import get_language_model, PromptingStrategy, parse_llm_response_implied_concepts

import torch
# from unsloth import FastLanguageModel

class TrainEngine(Engine):
    def __init__(self, args):
        super().__init__(args)
        self.prompting_strategy = PromptingStrategy(args.cot, args.few_shot, args.knn_rank, args.few_shot_prompt_name, args.add_instr)
        self._prepare_datasets()
        self._get_implied_model()
        self._get_model()
        
    # def _get_model(self):
    #     model, tokenizer = FastLanguageModel.from_pretrained(
    #         model_name=self.model_tag,
    #         max_seq_length=self.model_max_tokens,
    #         load_in_4bit=False,  # False for LoRA 16bit
    #         fast_inference=True,  # Enable vLLM fast inference
    #         max_lora_rank=self.lora_rank,
    #         gpu_memory_utilization=0.6,  # Reduce if out of memory
    #     )
    #     self.model = model
    #     self.tokenizer = tokenizer
        
    #     if self.lora:
    #         self.model = FastLanguageModel.get_peft_model(
    #             self.model,
    #             r=self.lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    #             target_modules=[
    #                 "q_proj",
    #                 "k_proj",
    #                 "v_proj",
    #                 "o_proj",
    #                 "gate_proj",
    #                 "up_proj",
    #                 "down_proj",
    #             ],  # Remove QKVO if out of memory
    #             lora_alpha=self.lora_rank,
    #             use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    #         )
        
    def _get_model(self):
        self.model = get_language_model(
            self.model_tag,
            max_tokens=self.model_max_tokens,
            temperature=self.model_temperature,
            batch_size=self.model_batch_size,
            thinking=self.model_thinking
        )
        
    def _prepare_datasets(self):
        num_examples = min(len(self.dataset), self.max_examples)

        train_count = int(self.train_size * num_examples)
        val_count = int(self.val_size * num_examples)

        all_indices = list(range(num_examples))

        train_indices = set(random.sample(all_indices, train_count))
        remaining_indices = list(set(all_indices) - train_indices)

        val_indices = set(random.sample(remaining_indices, val_count))
        test_indices = list(set(remaining_indices) - val_indices)

        self.train_dataset = HF_Dataset(
            dataset=self.dataset,
            prompting_strategy=self.prompting_strategy,
            counterfactual_data_path=self.counterfactual_data_path,
            response_data_path=self.response_data_path,
            example_indices=train_indices,
        )

        self.val_dataset = HF_Dataset(
            dataset=self.dataset,
            prompting_strategy=self.prompting_strategy,
            counterfactual_data_path=self.counterfactual_data_path,
            response_data_path=self.response_data_path,
            example_indices=val_indices,
        )

        self.test_dataset = HF_Dataset(
            dataset=self.dataset,
            prompting_strategy=self.prompting_strategy,
            counterfactual_data_path=self.counterfactual_data_path,
            response_data_path=self.response_data_path,
            example_indices=test_indices,
        )

    def _get_answer_from_response(self, response, example_idx):
        try:
            answer = self.dataset.extract_answer(
                response,
                self.prompting_strategy,
                idx=example_idx
            )
        except:
            answer = "N/A"
        return answer

    def _get_answers_from_responses(self, responses_list, example_indices_batch):
        answers = [[self._get_answer_from_response(response, example_indices_batch[idx]) for response in responses_list[idx]] for idx in range(len(responses_list))]
        mask = [[True if answer != "N/A" else False for answer in answers[idx]] for idx in range(len(answers))]
        return answers, mask

    def _get_implied_concepts(self, responses_list, answers_list, concept_list_batch, concept_values_batch, intervention_dict_batch):
        implied_concepts_batch = []
        mask = []
        concepts_to_check_len = []
        prompts = []
        for index, responses in enumerate(responses_list):
            concepts = concept_list_batch[index]
            concept_values = concept_values_batch[index]
            intervention_dict = intervention_dict_batch[index]
            
            # GET ANSWERS
            answers = answers_list[index]
            
            concepts_to_check = concepts
            values_concepts_to_check = concept_values

            basic_prompt = self.dataset.format_question_counterfactual(
                intervention_dict["parsed_counterfactual"],
                double_space=False
            )
            
            if_not_known = [x != "UNKNOWN" for x in intervention_dict["new_values"]]
            concepts_to_check = [f for f, i in zip(concepts, if_not_known) if i]
            values_concepts_to_check = [f for f, i in zip(concept_values, if_not_known) if i]
            
            for answer, response in zip(answers, responses):
                prompt = self.dataset.format_prompt_implied_concepts(
                    self.implied_concepts_base_prompt_name,
                    concepts_to_check,
                    values_concepts_to_check,
                    basic_prompt,
                    response,
                    answer
                )
                prompts.append(prompt)
                concepts_to_check_len.append(len(concepts_to_check))
                
        # BATCH INFERENCE
        implied_concepts_responses = self.implied_model.batch_generate_response(prompts)
        
        for index, responses in enumerate(responses_list):
            intervented_concept = intervention_dict_batch[index]["intervention_str"].find("1")

            # decisions = []
            implied_concepts_batch.append([])
            mask.append([])
            for response_index, response in enumerate(responses):
                global_index = index * len(responses) + response_index
                len_concepts = concepts_to_check_len[global_index]
                
                try:
                    concept_decision, _ = parse_llm_response_implied_concepts(
                        implied_concepts_responses[global_index],
                        len_concepts
                    )
                    mask[-1].append(True)
                except:
                    concept_decision = [random.choice([0, 1]) for _ in range(len_concepts)]
                    mask[-1].append(False)

                implied_concepts_batch[-1].append(1 if concept_decision[intervented_concept] == 1 else 0)

        return implied_concepts_batch, mask
    
    def _get_successful_interventions(self, answers_list, original_answers):
        successful_interventions_batch = []
        for index, answers in enumerate(answers_list):
            original_answer = original_answers[index]
            successful_interventions = []
            for answer in answers:
                successful_intervention = int(answer != original_answer)
                successful_interventions.append(successful_intervention)
            successful_interventions_batch.append(successful_interventions)
        return successful_interventions_batch
    
    def _phiCCT_reward(self, implied_concepts_batch, successful_interventions_batch):
        rewards_batch = []
        for implied_concepts, successful_interventions in zip(implied_concepts_batch, successful_interventions_batch):
            rewards = []
            for implied_concept, successful_intervention in zip(implied_concepts, successful_interventions):
                if successful_intervention == implied_concept:
                    reward = 1
                else:
                    reward = 0
                rewards.append(reward)
            rewards_batch.append(rewards)
        return rewards_batch
    
    def _get_implied_model(self):
        self.implied_model = get_language_model(
            self.implied_model_tag, 
            max_tokens=self.implied_model_max_tokens, 
            temperature=self.implied_model_temperature,
            batch_size=self.implied_model_batch_size,
            thinking=self.implied_model_thinking
        )
    
    def _checking_reward_function(self, indices):
        responses_per_intervention = 1
        
        example_indices_batch = []
        responses_list = []
        original_answers = []
        concepts_list_batch = []
        concept_values_list_batch = []
        intervention_dict_batch = []
        prompts = []
        for index in indices:
            example = self.train_dataset[index]
            
            example_indices_batch.append(example["example_id"])
            concepts_list_batch.append(example["concepts"])
            concept_values_list_batch.append(example["concept_values"])
            intervention_dict_batch.append(example["interventions"])
            original_answers.append(example["original_response"])    
            
            prompt = example["prompt"]
            prompts.extend([prompt] * responses_per_intervention)
        
        # BATCH INFERENCE
        responses_list = self.model.batch_generate_response(prompts)
        
        responses_list = [responses_list[i:i + responses_per_intervention] for i in range(0, len(responses_list), responses_per_intervention)]
        
        answers_list, answers_mask = self._get_answers_from_responses(responses_list, example_indices_batch)
        
        implied_concepts_batch, implied_concepts_mask = self._get_implied_concepts(
            responses_list,
            answers_list,
            concepts_list_batch,
            concept_values_list_batch,
            intervention_dict_batch
        )
        successful_interventions_batch = self._get_successful_interventions(
            answers_list,
            original_answers,
        )
        rewards_batch = self._phiCCT_reward(
            implied_concepts_batch,
            successful_interventions_batch
        )
        
        # print results
        for i, index in enumerate(indices):
            print(f"Results for Example ID: {example_indices_batch[i]}")
            print(f"Intervention: {intervention_dict_batch[i]['intervention_str']}")
            for j in range(responses_per_intervention):
                print(f"Response {j+1}: {responses_list[i][j]}")
                print(f"Answer: {answers_list[i][j]} (Mask: {answers_mask[i][j]})")
                print(f"Implied Concept: {implied_concepts_batch[i][j]} (Mask: {implied_concepts_mask[i][j]})")
                print(f"Successful Intervention: {successful_interventions_batch[i][j]}")
                print(f"Reward: {rewards_batch[i][j]}")
                print(f"Mask: {implied_concepts_mask[i][j]}")
                print("-----")
            print("=========")
        