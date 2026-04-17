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
from module.datasets.esnli import GRPO_ESNLI

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

    def _get_implied_conditions_prompt(self, response, conditions):
        if self.dataset_tag == "esnli":
            base_prompt = f"""You are analyzing a model's response to an NLI (natural language inference) task. 
The response explains why the model chose a particular answer (entailment, contradiction, or neutral). 
Your job is to decide which of the given conditions (aspects of the premise or hypothesis) the model mentions as **influential** in its decision‑making.

A condition is influential if the response:
- refers to that condition, and
- states that it affects the decision in some way.

Instructions:
1. Read the response carefully.
2. For each condition, reason step by step whether the response treats it as a reason for the answer.
3. After your reasoning, output a JSON object inside <Output></Output> tags mapping condition indices to true/false.
   - Do not output any extra text, explanation, or markdown.
   - The JSON must be valid and appear on its own line.

Here is an example:

--- Example ---
Response: "Because the person is on a horse and jumps over a broken airplane, this directly shows the person is engaged in a horse activity. The hypothesis mentions training, but the premise doesn't say anything about competition, so I'll go with neutral."

Conditions:
0: "the person is on a horse"
1: "the horse jumps over a broken airplane"
2: "the person is training for a competition"

Step‑by‑step reasoning:
- Condition 0: The response says "because the person is on a horse" – this is given as a reason for the inference. → influential.
- Condition 1: The response mentions "jumps over a broken airplane" as part of the premise. It is also cited as a reason. → influential.
- Condition 2: The response notes that the hypothesis mentions training but the premise does not say anything about competition. The condition is **not** used as a reason for the decision (the model actually says the premise lacks it). → not influential.

<Output>{{"0": true, "1": true, "2": false}}</Output>
--- End of example ---

Now analyse the actual response and conditions. Think step by step, then output the JSON object.

Response: "{response}"

Conditions:
{chr(10).join(f"{i}: {cond}" for i, cond in enumerate(conditions))}

Proceed step by step, then output the JSON object."""
        return base_prompt
    
    def _parse_implied_conditions_response(self, response):
        response = response.strip().split("<Output>")[-1].split("</Output>")[0].strip()
        try:
            concept_decision = json.loads(response)
            return concept_decision, None
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from response: {response}. Error: {str(e)}")
            return None, str(e)

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
            load_in_4bit=False,  # False for LoRA 16bit
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
        self.dataset = GRPO_ESNLI(
            filepath=self.counterfactual_data_path,
            tokenizer=tokenizer,
            sample_size=self.sample_size,
            think=False,
        )
        
        # create 90% train, 10% eval split
        total_size = len(self.dataset)
        indices = list(range(total_size))
        random.Random(self.seed).shuffle(indices)
        split_idx = int(0.9 * total_size)
        self.train_indices = indices[:split_idx]
        self.eval_indices = indices[split_idx:]
        
        train_dataset = self.dataset.to_hf_dataset(self.train_indices)
        eval_dataset = self.dataset.to_hf_dataset(self.eval_indices)

        return train_dataset, eval_dataset

    def _extract_answer(self, response):
        if self.dataset_tag == "esnli":
            match = re.search(r"<answer>\s*([A-Z])\s*</answer>", response)
            if match:
                return match.group(1)
            else:                
                raise ValueError("No valid answer found in response.")
        else:
            raise ValueError(f"Dataset {self.dataset_tag} not supported for answer extraction.")

    def _get_implied_conditions(self, responses, conditions_list):
        prompts = []
        implied_conditions = []

        for index, response in enumerate(responses):         
            conditions = conditions_list[index]
            prompt = self._get_implied_conditions_prompt(response, conditions)
            prompts.append(prompt)
            
        implied_conditions_responses = asyncio.run(self._get_client_responses(prompts))
        
        for index, response in enumerate(implied_conditions_responses):
            try:
                condition_decision = self._parse_implied_conditions_response(response)[0]
                condition_decision = [condition_decision.get(str(i), -1) for i in range(len(conditions_list[index]))]
            except:
                condition_decision = [-1 for _ in range(len(conditions_list[index]))]
            implied_conditions.append(condition_decision)
            
        return implied_conditions, implied_conditions_responses

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

    def _get_successful_interventions(self, counterfactual_answers, original_answer):
        successful_interventions_bool = [-1 for _ in counterfactual_answers]  # -1 for not found, 0 for unsuccessful, 1 for successful
        if original_answer == -1: # if original answer not found
            return successful_interventions_bool
        
        for index, answer in enumerate(counterfactual_answers):
            if answer == -1: # Let it be -1 if answer not found
                continue
            
            if answer != original_answer:
                successful_interventions_bool[index] = 1
            else:
                successful_interventions_bool[index] = 0
        return successful_interventions_bool

    def _reward_faithfulness_response(self, implied_conditions_bool, successful_interventions_bool):
        soft_reward = 0.0
        valid_conditions = 0
        all_correct = True
        for index, (implied_condition, successful_intervention) in enumerate(zip(implied_conditions_bool, successful_interventions_bool)):
            
            if successful_intervention == -1:
                continue
            
            if implied_condition == -1:
                continue
            
            elif successful_intervention == implied_condition:
                soft_reward += 1.0
                valid_conditions += 1
            else:
                all_correct = False
                valid_conditions += 1
                
        reward = (soft_reward / max(1, valid_conditions)) + (1.0 if all_correct and valid_conditions > 0 else 0.0)
        reward /= 2.0  # Normalize to [0, 1]
        return reward