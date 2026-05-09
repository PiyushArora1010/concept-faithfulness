import json
import numpy as np
from datasets import Dataset, concatenate_datasets
import random

class ESNLI:
    def __init__(self, filepath: str, split="train", sample_size=None):
        self.filepath = filepath
        self.split = split
        self.sample_size = sample_size
        
        self.labels = ['A', 'B']
        
        self.data = self._load()
        self.indices = list(range(len(self.data)))
        
        if self.sample_size is not None and self.sample_size < len(self.data):
            indices = np.random.permutation(len(self.data))[:self.sample_size].tolist()
            self.indices = indices
            self.data = [self.data[i] for i in indices]

    def _label_map(self, label):
        return {
            'entailment': 'A',
            'contradiction': 'B',
        }.get(label, None)

    def _format_question(self, s1, s2):
        return f'Context: {s1}\nQuestion: Can we infer that {s2}?'
    
    def _process(self, dataset: Dataset):
        processed = []
        for idx, ex in enumerate(dataset):
            correct = self._label_map(ex['gold_label'])
            if correct is None:
                continue
            s1, s2 = ex['Sentence1'], ex['Sentence2']
            if s2 is None or s1 is None:
                continue
            if len(s1.split()) < 3 or len(s2.split()) < 3:
                continue
            
            question_text = self._format_question(s1, s2)
            answer_map = {
                'A': 'Yes',
                'B': 'No'
            }

            answer_choices = "(A) Yes\n(B) No"
            
            processed.append({
                "example_idx": idx,
                "question": question_text,
                "answer_choices": answer_choices,
                "label": correct
            })
        return processed

    def _load(self):
        if self.split == "train":
            d1 = Dataset.from_csv(f"{self.filepath}/dataset/esnli_train_1.csv")
            d2 = Dataset.from_csv(f"{self.filepath}/dataset/esnli_train_2.csv")
            merged = concatenate_datasets([d1, d2], axis=0)
            return self._process(merged)
        elif self.split == "test":
            test = Dataset.from_csv(f"{self.filepath}/dataset/esnli_test.csv")
            return self._process(test)
        else:
            raise ValueError("split must be 'train' or 'test'")

    def _create_question_with_choices(self, idx):
        ex = self.data[idx]
        return f"{ex['question']}\nAnswer Choices:\n{ex['answer_choices']}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class GRPO_ESNLI:
    def __init__(self, filepath, tokenizer, sample_size=None, question_wrapper=None, think=False, engine=None):
        self.tokenizer = tokenizer
        if question_wrapper is not None:
            self.question_wrapper = question_wrapper
        else:
            self.question_wrapper = """{question}\nThink step by step, then give your final answer inside <answer></answer> tags. Your answer should be a single letter (A, B, or C)."""

        self.think = think
        self.add_generation_prompt = True
        self.answer_choices = "(A) Yes\n(B) No"
        self.data = self._load_jsonl(filepath)
        self.indices = list(range(len(self.data)))
        
        if sample_size is not None and sample_size < len(self.data):
            rng = random.Random(0)
            indices = rng.sample(range(len(self.data)), sample_size)
            self.indices = indices
            self.data = [self.data[i] for i in indices]
        
        self._prepare_prompts(engine=engine)

    def _load_jsonl(self, filepath):
        data = []
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return data

    def _apply_chat_template(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=self.add_generation_prompt,
            enable_thinking=self.think
        )

    def _format_question_with_choices(self, context, question, apply_wrapper=True, answer_choices=True):
        context_question = f"Context: {context}\nQuestion: {question}"
        if answer_choices:
            question = f"{context_question}\nAnswer Choices:\n{self.answer_choices}"
        else:
            question = context_question
        if apply_wrapper:
            question = self.question_wrapper.format(question=question)
        return question

    def _prepare_prompts(self, engine=None):
        len_data = len(self.data)
        data_new = []
        for item in self.data:
            
            original_full = self._format_question_with_choices(item["original_context"], item["original_question"])
            original_question_without_wrapper = self._format_question_with_choices(item["original_context"], item["original_question"], apply_wrapper=False)
            item["prompt"] = self._apply_chat_template(original_full)
            item["question"] = original_question_without_wrapper

            # Counterfactuals
            cf_prompts = []
            original_conditions = []
            if len(item["counterfactuals"]) == 0:
                continue
            for cf in item.get("counterfactuals", []):
                
                context = cf.get("counterfactual_context", "")
                question = item.get("original_question", "") # use original question

                cf_full = self._format_question_with_choices(context, question, apply_wrapper=True)
                cf_prompts.append(self._apply_chat_template(cf_full))
                original_conditions.append(cf.get("original_value"))
                
            item["counterfactual_prompts"] = cf_prompts
            item["original_conditions"] = original_conditions
            
            data_new.append(item)
            
            if engine is not None and engine.hint_cf:
                hint_item = {}
                hint_types_available = engine.train_hint_types
                random_hint_type = random.choice(hint_types_available)
                hinted_answer = random.choice(['A', 'B'])
                variants = engine._generate_prompt_variants(
                    random_hint_type,
                    self._format_question_with_choices(item["original_context"], item["original_question"], apply_wrapper=False),
                    hinted_answer
                )
                choosen_variant_idx = random.randint(0, len(variants)-1)
                choosen_variant = variants[choosen_variant_idx]
                hint_item["gt"] = item["gt"]
                hint_item["prompt"] = self._apply_chat_template(self.question_wrapper.format(question=choosen_variant))
                hint_item["counterfactual_prompts"] = [
                    item["prompt"]
                ]
                
                for cf in item.get("counterfactuals", []):
                    context_cf = cf.get("counterfactual_context", "")
                    question_cf = item.get("original_question", "") # use original question
                    
                    variants_cf = engine._generate_prompt_variants(
                        random_hint_type,
                        self._format_question_with_choices(context_cf, question_cf, apply_wrapper=False),
                        hinted_answer
                    )
                    choosen_variant_cf = variants_cf[choosen_variant_idx]
                    hint_item["counterfactual_prompts"].append(self._apply_chat_template(self.question_wrapper.format(question=choosen_variant_cf)))
                
                hint_item["example_idx"] = len_data + item["example_idx"]
                hint_item["original_conditions"] = [engine._generate_prompt_variants(
                    random_hint_type,
                    "",
                    hinted_answer
                )[choosen_variant_idx].strip()] + item["original_conditions"]
                hint_item["question"] = choosen_variant
                data_new.append(hint_item)
            
        self.data = data_new
        
        # print one example for debugging
        if len(self.data) > 0:
            example = self.data[-1]
            
            print(f"Prompt:\n{example['prompt']}\n")
            for condition, cf_prompt in zip(example["original_conditions"], example["counterfactual_prompts"]):
                print(f"Condition: {condition}\nCounterfactual Prompt:\n{cf_prompt}\n")
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "example_id": item["example_idx"],
            "gt": item["gt"],
            "prompt": item["prompt"],
            "question": item["question"],
            "counterfactual_prompts": item["counterfactual_prompts"],
            "original_conditions": item["original_conditions"],
        }

    def to_hf_dataset(self, indices=None):
        if indices is None:
            indices = list(range(len(self)))
        return Dataset.from_list([self[i] for i in indices])


class GRPO_ESNLI_CF:
    def __init__(self, filepath, tokenizer, sample_size=None, question_wrapper=None, think=False, engine=None):
        self.tokenizer = tokenizer
        if question_wrapper is not None:
            self.question_wrapper = question_wrapper
        else:
            self.question_wrapper = """{question}\nThink step by step, then give your final answer inside <answer></answer> tags. Your answer should be a single letter (A, B, or C)."""

        self.think = think
        self.add_generation_prompt = True
        self.answer_choices = "(A) Yes\n(B) No"
        self.label_map = {
            "A": "Yes",
            "B": "No"
        }
        self.labels = ['A', 'B']
        self.data = self._load_jsonl(filepath)
        self.indices = list(range(len(self.data)))
        
        if sample_size is not None and sample_size < len(self.data):
            rng = random.Random(0)
            indices = rng.sample(range(len(self.data)), sample_size)
            self.indices = indices
            self.data = [self.data[i] for i in indices]
        
        self._prepare_prompts(engine=engine)

    def _load_jsonl(self, filepath):
        data = []
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return data

    def _apply_chat_template(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=self.add_generation_prompt,
            enable_thinking=self.think
        )

    def _format_question_with_choices(self, context_item, question_item, apply_wrapper=True):
        context_question = f"Context: {context_item}\nQuestion: {question_item}"
        question = f"{context_question}\nAnswer Choices:\n{self.answer_choices}"
        if apply_wrapper:
            return self.question_wrapper.format(question=question)
        return question

    def _prepare_prompts(self, engine=None):
        data = []
        for item in self.data:

            original_question_prompt = self._apply_chat_template(
                self._format_question_with_choices(
                    item["original_context"], 
                    item["original_question"]
                )
            )

            if len(item["counterfactuals"]) == 0:
                continue
            
            for cf in item.get("counterfactuals", []):
                context = cf.get("counterfactual_context", "")
                question = item["original_question"] # replace counterfactual_question with original question
                
                cf_question_text = self._format_question_with_choices(context, question)
                cf_question_prompt = self._apply_chat_template(cf_question_text)
                
                data.append({
                    "example_id": item["example_idx"],
                    "original_prompt": original_question_prompt,
                    "prompt": cf_question_prompt,
                    "condition": cf.get("alternative_value"),
                    "question": self._format_question_with_choices(context, question, apply_wrapper=False),
                })
            
            if engine is not None:
                if not engine.hint_cf:
                    continue
                
                hint_types_available = engine.train_hint_types
                random_hint_type = random.choice(hint_types_available)
                
                hinted_answer = random.choice(self.labels)
                
                variants = engine._generate_prompt_variants(
                    random_hint_type,
                    self._format_question_with_choices(item["original_context"], item["original_question"], apply_wrapper=False),
                    hinted_answer
                )
                choosen_variant_idx = random.randint(0, len(variants)-1)
                choosen_variant = variants[choosen_variant_idx]
                choosen_variant_prompt = self._apply_chat_template(self.question_wrapper.format(question=choosen_variant))
                
                added_hint = engine._generate_prompt_variants(
                    random_hint_type,
                    "",
                    hinted_answer
                )[choosen_variant_idx].strip()
                
                data.append(
                    {
                        "example_id": item["example_idx"],
                        "original_prompt": original_question_prompt,
                        "prompt": choosen_variant_prompt,
                        "condition": added_hint,
                        "question": choosen_variant,
                    }
                )
      
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item

    def to_hf_dataset(self, indices=None):
        if indices is None:
            indices = list(range(len(self)))
        return Dataset.from_list([self[i] for i in indices])
    