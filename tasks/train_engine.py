import os
import asyncio
import sys
import random
from openai import AsyncOpenAI

from tasks.engine import Engine
from module.datasets.dataset import HF_Dataset
from module.utils import get_language_model, PromptingStrategy, parse_llm_response_implied_concepts

import torch
from unsloth import FastLanguageModel

class TrainEngine(Engine):
    def __init__(self, args):
        super().__init__(args)
        self.prompting_strategy = PromptingStrategy(args.cot, args.few_shot, args.knn_rank, args.few_shot_prompt_name, args.add_instr)
        self.implied_client = None
        self._get_implied_client()

    # PREPARE DATASETS
    def _prepare_datasets(self):
        num_examples = min(len(self.dataset), self.max_examples)

        train_count = int(self.train_size * num_examples)
        val_count = int(self.val_size * num_examples)

        all_indices = list(range(num_examples))

        train_indices = set(random.sample(all_indices, train_count))
        remaining_indices = list(set(all_indices) - train_indices)

        val_indices = set(random.sample(remaining_indices, val_count))
        test_indices = list(set(remaining_indices) - val_indices)

        train_dataset = HF_Dataset(
            dataset=self.dataset,
            prompting_strategy=self.prompting_strategy,
            counterfactual_data_path=self.counterfactual_data_path,
            response_data_path=self.response_data_path,
            example_indices=train_indices,
        )

        val_dataset = HF_Dataset(
            dataset=self.dataset,
            prompting_strategy=self.prompting_strategy,
            counterfactual_data_path=self.counterfactual_data_path,
            response_data_path=self.response_data_path,
            example_indices=val_indices,
        )

        test_dataset = HF_Dataset(
            dataset=self.dataset,
            prompting_strategy=self.prompting_strategy,
            counterfactual_data_path=self.counterfactual_data_path,
            response_data_path=self.response_data_path,
            example_indices=test_indices,
        )
        
        return train_dataset, val_dataset, test_dataset

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
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],  # Remove QKVO if out of memory
                lora_alpha=self.lora_rank,
                use_gradient_checkpointing="unsloth",  # Enable long context finetuning
            )
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

    # PROCESSING RESPONSES
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

    def _get_answers_from_responses(self, responses, example_indices):
        K = self.completions_per_prompt

        answers = []
        
        for index, response in enumerate(responses):
            outer_index = index // K
            example_idx = example_indices[outer_index]
            answer = self._get_answer_from_response(response, example_idx)
            answers.append(answer)
        
        mask = torch.tensor([True if answer != "N/A" else False for answer in answers])
        return answers, mask

    def _get_implied_concepts(self, responses, answers, concepts_list, concept_values_list, intervention_dict_list):
        K = self.completions_per_prompt
        
        concepts_to_check_len = []
        prompts = []
        implied_concepts = []

        for index, response in enumerate(responses):
            outer_index = index // K
            
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
            outer_index = index // K
            intervented_concept = intervention_dict_list[outer_index]["intervention_str"].find("1")
            len_concepts = concepts_to_check_len[index]
            try:
                concept_decision, _ = parse_llm_response_implied_concepts(
                    response,
                    len_concepts,
                )
                implied_concept = 1 if concept_decision[intervented_concept] == 1 else 0
            except:
                implied_concept = "N/A"
            implied_concepts.append(implied_concept)
        mask = torch.tensor([True if ic != "N/A" else False for ic in implied_concepts])
        return implied_concepts, mask

    def _get_successful_interventions(self, answers, original_answers):
        successful_interventions = []
        for index, answer in enumerate(answers):
            outer_index = index // self.completions_per_prompt
            original_answer = original_answers[outer_index]
            successful_intervention = int(answer != original_answer)
            successful_interventions.append(successful_intervention)
        return successful_interventions
    
    # REWARD FUNCTION (FAITHFULNESS)
    def _phiCCT(self, implied_concepts, successful_interventions, mask):
        rewards = []
        for index, (implied_concept, successful_intervention) in enumerate(zip(implied_concepts, successful_interventions)):
            
            if not mask[index]:
                rewards.append(0.0)
                continue
            
            if successful_intervention == implied_concept:
                reward = 1.0
            else:
                reward = 0.0
            rewards.append(reward)

        return rewards
