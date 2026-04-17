import json
import random
from datasets import Dataset, concatenate_datasets

class ESNLI:
    def __init__(self, filepath: str, split="train", sample_size=None):
        self.filepath = filepath
        self.split = split
        self.sample_size = sample_size
        self.labels = ['A', 'B', 'C']
        self.data = self._load()
        self.indices = list(range(len(self.data)))
        
        if self.sample_size is not None and self.sample_size < len(self.data):
            rng = random.Random(0)                     # fixed seed for reproducibility
            indices = rng.sample(range(len(self.data)), self.sample_size)   # choose random indices
            self.indices = indices
            self.data = [self.data[i] for i in indices]                     # extract subset

    def _label_map(self, label):
        return {
            'entailment': 'A',
            'contradiction': 'B',
            'neutral': 'C'
        }.get(label, None)

    def _format_question(self, s1, s2):
        # Include answer choices in the prompt
        return f'Context: {s1}\nQuestion: Can we infer that {s2}?'
    
    def _process(self, dataset: Dataset):
        processed = []
        for ex in dataset:
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
                'B': 'No',
                'C': 'Maybe, this is neutral'
            }

            # Also store answer choices separately for convenience
            answer_choices = "(A) Yes\n(B) No\n(C) Maybe, this is neutral"
            
            processed.append({
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
    def __init__(self, filepath, tokenizer, sample_size=None, think=False):
        self.tokenizer = tokenizer
        self.question_wrapper = """{question}\nThink step by step, then give your final answer inside <answer></answer> tags. Your answer should be a single letter (A, B, or C)."""

        self.think = think
        self.add_generation_prompt = True
        self.answer_choices = "(A) Yes\n(B) No\n(C) Maybe, this is neutral"
        self.data = self._load_jsonl(filepath)
        self.indices = list(range(len(self.data)))
        
        if sample_size is not None and sample_size < len(self.data):
            rng = random.Random(0)
            indices = rng.sample(range(len(self.data)), sample_size)
            self.indices = indices
            self.data = [self.data[i] for i in indices]
        
        self._prepare_prompts()

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

    def _format_question_with_choices(self, item):
        context_question = f"Context: {item['original_context']}\nQuestion: {item['original_question']}"
        question = f"{context_question}\nAnswer Choices:\n{self.answer_choices}"
        return self.question_wrapper.format(question=question)

    def _prepare_prompts(self):
        for item in self.data:
            # Original prompt with answer choices
            original_full = self._format_question_with_choices(item)
            item["prompt"] = self._apply_chat_template(original_full)

            # Counterfactuals
            cf_prompts = []
            cf_metadata = []
            original_conditions = []
            if len(item["counterfactuals"]) == 0:
                continue
            for cf in item.get("counterfactuals", []):
                # Reconstruct full question from counterfactual context + question
                context = cf.get("counterfactual_context", "")
                question = cf.get("counterfactual_question", "")
                if not context.startswith("Context: "):
                    context = f"Context: {context}"
                if not question.startswith("Question: "):
                    question = f"Question: {question}"
                cf_question_text = f"{context}\n{question}"
                cf_full = f"{cf_question_text}\nAnswer Choices:\n{self.answer_choices}"
                cf_full = self.question_wrapper.format(question=cf_full)
                cf_prompts.append(self._apply_chat_template(cf_full))
                cf_metadata.append({
                    "condition_index": cf.get("condition_index"),
                    "original_value": cf.get("original_value"),
                    "alternative_value": cf.get("alternative_value"),
                    "counterfactual_context": cf.get("counterfactual_context"),
                    "counterfactual_question": cf.get("counterfactual_question"),
                })
                original_conditions.append(cf.get("original_value"))
                
            item["counterfactual_prompts"] = cf_prompts
            item["counterfactual_metadata"] = cf_metadata
            item["original_conditions"] = original_conditions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "example_id": item["example_idx"],
            "prompt": item["prompt"],
            "counterfactual_prompts": item["counterfactual_prompts"],
            "counterfactual_metadata": item["counterfactual_metadata"],
            "original_conditions": item["original_conditions"],
        }

    def to_hf_dataset(self, indices=None):
        if indices is None:
            indices = list(range(len(self)))
        return Dataset.from_list([self[i] for i in indices])
    