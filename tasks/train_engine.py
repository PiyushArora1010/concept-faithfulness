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

import torch
from unsloth import FastLanguageModel
from trl import GRPOTrainer
    
class TrainEngine(Engine):
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
        pass

    def _extract_answer(self, response):
        return response.split("<answer>")[-1].split("</answer>")[0].strip()

    def _get_implied_concepts(self, responses, answers, conditions_list, example_indices):
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

    def _get_answers_from_responses(self, responses):
        answers = []
        for response in responses:
            try:
                answer = self._extract_answer(response)
            except:
                answer = -1
            answers.append(answer)
        return answers
                

    def _get_counterfactual_answers(self, model, prompts_list):
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self.model_max_tokens,
        )
        # shape of prompts_list: List[List[str]]
        
        # Flatten the list of lists and keep track of original structure
        flat_prompts = []
        structure = []  # Store the length of each sublist
        
        for sublist in prompts_list:
            flat_prompts.extend(sublist)
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
                    try:
                        answer = self._extract_answer(response)
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