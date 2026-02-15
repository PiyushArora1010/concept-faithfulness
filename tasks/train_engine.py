import os
import json
import asyncio
import sys
import re
import random
from string import ascii_uppercase
from vllm import SamplingParams
from openai import AsyncOpenAI

from tasks.engine import Engine
from module.datasets.dataset import GRPODataset, ConditionsGRPODataset
from module.utils import get_language_model, PromptingStrategy, parse_llm_response_implied_concepts

import torch
from unsloth import FastLanguageModel
from trl import GRPOTrainer

# class DecisionMaskedTrainerGRPO(GRPOTrainer):
#     def __init__(self, *args, engine, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.engine = engine
    
#     def _get_decision_index(self, completion_text, processing_class):
#         answer_found = self.engine.dataset.answer_starting_index(completion_text, self.engine.prompting_strategy)

#         if answer_found < 0:
#             return [], []

#         encoding = processing_class(
#             completion_text,
#             add_special_tokens=False,
#             return_offsets_mapping=True,
#         )

#         offsets = encoding["offset_mapping"]
#         full_ids = encoding["input_ids"]

#         answer_token_start = -1
#         for i, (start, end) in enumerate(offsets):
#             if start <= answer_found < end:
#                 answer_token_start = i
#                 break

#         if answer_token_start == -1:
#             return [], []

#         explanation_indices = list(range(0, answer_token_start))
#         decision_indices = list(range(answer_token_start, len(full_ids)))

#         return explanation_indices, decision_indices
    
#     def _generate_and_score_completions(self, inputs):
#         outputs = super()._generate_and_score_completions(inputs)
#         if self.engine.loss_computed_on == "both":
#             return outputs

#         completion_ids = outputs["completion_ids"]
#         completion_mask = outputs["completion_mask"]

#         completions_text = self.processing_class.batch_decode(
#             completion_ids, skip_special_tokens=True
#         )

#         custom_masks = []

#         for i, (comp_text, comp_ids) in enumerate(zip(completions_text, completion_ids)):

#             explanation_indices, decision_indices = self._get_decision_index(
#                 comp_text, self.processing_class
#             )
            
#             explanation_indices = [index for index in explanation_indices if index < len(comp_ids)]
#             decision_indices = [index for index in decision_indices if index < len(comp_ids)]
            
#             custom_mask = torch.zeros_like(comp_ids, dtype=torch.int)

#             if decision_indices and explanation_indices:
#                 if self.engine.loss_computed_on == "decision":
#                     custom_mask[decision_indices] = 1

#                 elif self.engine.loss_computed_on == "explanation":
#                     custom_mask[explanation_indices] = 1
                    
#                 else:
#                     raise ValueError(f"Invalid loss_computed_on: {self.engine.loss_computed_on}")

#             custom_masks.append(custom_mask)

#         decision_explanation_mask = torch.stack(custom_masks).to(completion_ids.device)

#         outputs["completion_mask"] = completion_mask * decision_explanation_mask
        
#         if self.engine.debug:
#             final_mask = outputs["completion_mask"]
#             print("\n[DEBUG] ===== GRPO TOKEN MASKING =====")
#             for i in range(min(2, final_mask.shape[0])):
#                 mask = final_mask[i]
                
#                 print(f"\n[DEBUG] Sample {i}:")
                
#                 # Get tokens being updated
#                 updated_indices = [idx for idx in range(len(mask)) if mask[idx] == 1]
#                 if updated_indices:
#                     updated_ids = completion_ids[i][updated_indices]
#                     updated_text = self.processing_class.decode(updated_ids, skip_special_tokens=False)
#                     print(f"[DEBUG] Text being updated (mask=1):")
#                     print(f"{updated_text}")
                
#                 # Get tokens NOT being updated
#                 not_updated_indices = [idx for idx in range(len(mask)) if mask[idx] == 0]
#                 if not_updated_indices:
#                     not_updated_ids = completion_ids[i][not_updated_indices]
#                     not_updated_text = self.processing_class.decode(not_updated_ids, skip_special_tokens=False)
#                     print(f"\n[DEBUG] Text NOT being updated (mask=0):")
#                     print(f"{not_updated_text}")
        
#         return outputs

class TrainEngineGRPO(Engine):
    def __init__(self, args):
        super().__init__(args)
        self.output_dir = os.path.join(
            "results",
            "training",
            self.dataset_tag,
            "grpo",
            args.output_dir
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.prompting_strategy = PromptingStrategy(args.cot, args.few_shot, args.knn_rank, args.few_shot_prompt_name, args.add_instr)
        self.implied_client = None
        self._get_implied_client()

    # PREPARE DATASETS
    def _prepare_datasets(self, tokenizer):
        num_examples = min(len(self.dataset), self.max_examples)

        all_indices = list(range(num_examples))
        random.shuffle(all_indices)
        
        val_indices = set(self.example_indices)
        train_indices = set(all_indices) - val_indices
        
        file_name = os.path.join(self.output_dir, "data_splits.json")
        with open(file_name, "w") as f:
            json.dump({
                "train_indices": list(train_indices),
                "val_indices": list(val_indices),
            }, f, indent=4)

        train_dataset = GRPODataset(
            dataset=self.dataset,
            prompting_strategy=self.prompting_strategy,
            counterfactual_data_path=self.counterfactual_data_path,
            example_indices=train_indices,
            tokenizer=tokenizer
        )

        val_dataset = GRPODataset(
            dataset=self.dataset,
            prompting_strategy=self.prompting_strategy,
            counterfactual_data_path=self.counterfactual_data_path,
            example_indices=val_indices,
            tokenizer=tokenizer
        )
        
        return train_dataset, val_dataset

    # MAIN MODEL
    def _get_model_and_tokenizer(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_tag,
            max_seq_length=self.model_max_tokens,
            load_in_4bit=False,  # False for LoRA 16bit
            fast_inference=True,  # Enable vLLM fast inference
            max_lora_rank=self.lora_rank,
            gpu_memory_utilization=0.4,  # Reduce if out of memory
        )
        
        if self.lora:
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules=self.lora_layers,  # Remove QKVO if out of memory
                lora_alpha=self.lora_rank*2,
                use_gradient_checkpointing="unsloth",  # Enable long context finetuning
                random_state=self.seed
            )
        model.generation_config.max_length = self.model_max_tokens
        model.generation_config.temperature = self.model_temperature
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    # IMPLIED CONCEPTS MODEL
    def _get_implied_client(self):
        self.implied_client = AsyncOpenAI(
            api_key="EMPTY",  # vLLM ignores this
            base_url=self.implied_model_url,
        )
    
    async def _get_client_responses(self, prompts):
        async def _call(prompt):
            resp = await self.implied_client.chat.completions.create(
                model=self.implied_model_tag,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.implied_model_max_tokens,
                temperature=0,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": self.implied_model_thinking
                    }
                },
                n=1,
            )
            return resp.choices[0].message.content

        tasks = [_call(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    # GET ORIGINAL ANSWERS
    def _get_original_answers(self, model, prompts, example_indices):
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self.model_max_tokens,
        )
        answers = []
        responses = []
        
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(prompts), self.model_batch_size):
                batch_end = min(i + self.model_batch_size, len(prompts))
                batch_prompts = prompts[i:batch_end]
                batch_indices = example_indices[i:batch_end]
                
                outputs = model.fast_generate(
                    batch_prompts,
                    sampling_params=sampling_params,
                )
                outputs = [output.outputs[0].text for output in outputs]
                
                # Extract answers from batch
                for batch_idx, response in enumerate(outputs):
                    corresponding_example_idx = batch_indices[batch_idx]
                    try:
                        answer = self.dataset.extract_answer(
                            response,
                            self.prompting_strategy,
                            idx=corresponding_example_idx
                        )
                    except:
                        answer = -1
                    answers.append(answer)
                    responses.append(response)
        
        return answers, responses
        
    # PROCESSING RESPONSES
    def _get_answer_from_response(self, response, example_idx):
        try:
            answer = self.dataset.extract_answer(
                response,
                self.prompting_strategy,
                idx=example_idx
            )
        except:
            answer = -1
        return answer

    def _get_answers_from_responses(self, responses, example_indices):
        answers = [self._get_answer_from_response(response, example_idx) for response, example_idx in zip(responses, example_indices)]
        return answers

    def _get_implied_concepts(self, responses, answers, concepts_list, concept_values_list, intervention_dict_list):
        concepts_to_check_len = []
        prompts = []
        implied_concepts = []

        for index, response in enumerate(responses):
            outer_index = index
            
            concepts = concepts_list[outer_index]
            concept_values = concept_values_list[outer_index]
            intervention_dict = intervention_dict_list[outer_index]
            answer = answers[index]

            basic_prompt = self.dataset.format_question_counterfactual(
                intervention_dict["parsed_counterfactual"],
                double_space=False,
            )

            prompt = self.dataset.format_prompt_implied_concepts(
                self.implied_concepts_base_prompt_name,
                concepts,
                concept_values,
                basic_prompt,
                response,
                answer,
            )
            prompts.append(prompt)
            concepts_to_check_len.append(len(concepts))
            
        implied_concepts_responses = asyncio.run(self._get_client_responses(prompts))
        
        for index, response in enumerate(implied_concepts_responses):
            outer_index = index
            intervented_concept = intervention_dict_list[outer_index]["intervention_str"].find("1")
            len_concepts = concepts_to_check_len[index]
            try:
                concept_decision, _ = parse_llm_response_implied_concepts(
                    response,
                    len_concepts,
                )
                implied_concept = 1 if concept_decision[intervented_concept] == 1 else 0
            except:
                implied_concept = -1
            implied_concepts.append(implied_concept)
            
        return implied_concepts, implied_concepts_responses

    def _get_successful_interventions(self, answers, original_answers):
        successful_interventions = []
        for index, answer in enumerate(answers):
            outer_index = index
            original_answer = original_answers[outer_index]
            successful_intervention = int(answer != original_answer)
            
            if original_answer == -1 or answer == -1:
                successful_intervention = -1
            
            successful_interventions.append(successful_intervention)
        return successful_interventions
    
    # REWARD FUNCTION (FAITHFULNESS)
    def _phiCCT(self, implied_concepts, successful_interventions):
        rewards = []
        for index, (implied_concept, successful_intervention) in enumerate(zip(implied_concepts, successful_interventions)):
            
            if successful_intervention == -1 or implied_concept == -1:
                reward = 0.0
            elif successful_intervention == implied_concept:
                reward = 1.0
            else:
                reward = -1.0
            rewards.append(reward)

        return rewards
    
class TrainEngineConditionsGRPO(Engine):
    def __init__(self, args):
        super().__init__(args)
        self.output_dir = os.path.join(
            "results",
            "training",
            self.dataset_tag,
            "grpo_conditions",
            args.output_dir
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.prompting_strategy = PromptingStrategy(args.cot, args.few_shot, args.knn_rank, args.few_shot_prompt_name, args.add_instr)
        self.implied_client = None
        self._get_implied_client()

    def _get_implied_client(self):
        self.implied_client = AsyncOpenAI(
            api_key="EMPTY",  # vLLM ignores this
            base_url=self.implied_model_url,
        )

    async def _get_client_responses(self, prompts):
        async def _call(prompt):
            resp = await self.implied_client.chat.completions.create(
                model=self.implied_model_tag,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.implied_model_max_tokens,
                temperature=0,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": self.implied_model_thinking
                    }
                },
                n=1,
            )
            return resp.choices[0].message.content

        tasks = [_call(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    # MAIN MODEL
    def _get_model_and_tokenizer(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_tag,
            max_seq_length=self.model_max_tokens,
            load_in_4bit=True,  # False for LoRA 16bit
            fast_inference=True,  # Enable vLLM fast inference
            max_lora_rank=self.lora_rank,
            gpu_memory_utilization=0.8,  # Reduce if out of memory
        )
        
        if self.lora:
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules=self.lora_layers,  # Remove QKVO if out of memory
                lora_alpha=self.lora_rank*2,
                use_gradient_checkpointing="unsloth",  # Enable long context finetuning
                random_state=self.seed
            )
        model.generation_config.max_length = self.model_max_tokens
        model.generation_config.temperature = self.model_temperature
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer
    
    def _prepare_datasets(self, tokenizer):
        num_examples = min(len(self.dataset), self.max_examples)

        all_indices = list(range(num_examples))
        random.shuffle(all_indices)
        
        val_indices = set(self.example_indices)
        train_indices = set(all_indices) - val_indices
        
        file_name = os.path.join(self.output_dir, "data_splits.json")
        with open(file_name, "w") as f:
            json.dump({
                "train_indices": list(train_indices),
                "val_indices": list(val_indices),
            }, f, indent=4)

        train_dataset = ConditionsGRPODataset(
            dataset=self.dataset,
            prompting_strategy=self.prompting_strategy,
            counterfactual_data_path=self.counterfactual_data_path,
            example_indices=train_indices,
            tokenizer=tokenizer,
        )

        val_dataset = ConditionsGRPODataset(
            dataset=self.dataset,
            prompting_strategy=self.prompting_strategy,
            counterfactual_data_path=self.counterfactual_data_path,
            example_indices=val_indices,
            tokenizer=tokenizer,
        )

        return train_dataset, val_dataset

    def _get_implied_concepts(self, responses, answers, concepts_list, concept_values_list, example_indices):
        concepts_to_check_len = []
        prompts = []
        implied_concepts = []

        for index, response in enumerate(responses):         
            example_idx = example_indices[index]   
            concepts = concepts_list[index]
            concept_values = concept_values_list[index]
            answer = answers[index]

            basic_prompt = self.dataset.format_prompt_basic(
                example_idx,
                double_space=False
            )

            prompt = self.dataset.format_prompt_implied_concepts(
                self.implied_concepts_base_prompt_name,
                concepts,
                concept_values,
                basic_prompt,
                response,
                answer,
            )
            prompts.append(prompt)
            concepts_to_check_len.append(len(concepts))
            
        implied_concepts_responses = asyncio.run(self._get_client_responses(prompts))
        
        for index, response in enumerate(implied_concepts_responses):
            len_concepts = concepts_to_check_len[index]
            try:
                concept_decision, _ = parse_llm_response_implied_concepts(
                    response,
                    len_concepts,
                )
                assert len(concept_decision) == len_concepts
            except:
                concept_decision = [-1] * len_concepts
            implied_concepts.append(concept_decision)
            
        return implied_concepts, implied_concepts_responses

    def _get_answers_from_responses(self, responses, example_indices):
        answers = []
        for example_index, response in zip(example_indices, responses):
            try:
                answer = self.dataset.extract_answer(
                    response,
                    self.prompting_strategy,
                    idx=example_index
                )
            except:
                answer = -1
            answers.append(answer)
        return answers
                

    def _get_counterfactual_answers(self, model, prompts_list, example_indices):
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self.model_max_tokens,
        )
        # shape of prompts_list: List[List[str]]
        # Example Indices: List[int]
        
        # Flatten the list of lists and keep track of original structure
        flat_prompts = []
        flat_example_indices = []  # Track which example_idx each prompt belongs to
        structure = []  # Store the length of each sublist
        
        for example_idx, sublist in zip(example_indices, prompts_list):
            flat_prompts.extend(sublist)
            # Each prompt in the sublist maps to the same example_idx
            flat_example_indices.extend([example_idx] * len(sublist))
            structure.append(len(sublist))
        
        # Initialize answers for all prompts
        flat_answers = []
        
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(flat_prompts), self.model_batch_size):
                batch_end = min(i + self.model_batch_size, len(flat_prompts))
                batch_prompts = flat_prompts[i:batch_end]
                
                outputs = model.fast_generate(
                    batch_prompts,
                    sampling_params=sampling_params,
                )
                outputs = [output.outputs[0].text for output in outputs]
                
                # Extract answers from batch
                for batch_idx, response in enumerate(outputs):
                    global_idx = i + batch_idx
                    corresponding_example_idx = flat_example_indices[global_idx]
                    try:
                        answer = self.dataset.extract_answer(
                            response,
                            self.prompting_strategy,
                            idx=corresponding_example_idx  # Use the actual example_idx
                        )
                    except:
                        answer = -1
                    flat_answers.append(answer)
        
        # Reconstruct the list of lists structure
        answers_list = []
        start_idx = 0
        for length in structure:
            answers_list.append(flat_answers[start_idx:start_idx + length])
            start_idx += length
        
        return answers_list

    def _get_successful_interventions(self, counterfactual_answers, counterfactual_intervention_strings, total_concepts, original_answer):
        successful_interventions_bool = [-1 for _ in range(total_concepts)]
        if original_answer == -1: # if original answer not found
            return successful_interventions_bool
        
        for index, answer in enumerate(counterfactual_answers):
            if answer == -1: # Let it be -1 if answer not found
                continue
            intervented_concept = counterfactual_intervention_strings[index].find("1")
            if answer != original_answer:
                successful_interventions_bool[intervented_concept] = 1
            else:
                successful_interventions_bool[intervented_concept] = 0
        return successful_interventions_bool

    def _reward_faithfulness_response(self, implied_conditions_bool, successful_interventions_bool):
        soft_reward = 0.0
        valid_concepts = 0
        all_correct = True
        for index, (implied_condition, successful_intervention) in enumerate(zip(implied_conditions_bool, successful_interventions_bool)):
            
            if successful_intervention == -1:
                continue
            
            if implied_condition == -1:
                continue
            
            elif successful_intervention == implied_condition:
                soft_reward += 1.0
                valid_concepts += 1
            else:
                all_correct = False
                valid_concepts += 1
                
        reward = (soft_reward / max(1, valid_concepts)) + (1.0 if all_correct and valid_concepts > 0 else 0.0)
        reward /= 2.0  # Normalize to [0, 1]
        return reward