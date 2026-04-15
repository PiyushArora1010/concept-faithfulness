from datasets import Dataset, concatenate_datasets

class ESNLI:
    def __init__(self, filepath: str, split="train"):
        self.filepath = filepath
        self.split = split
        self.labels = ['A', 'B', 'C']
        self.data = self._load()

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
        # return len(self.data)
        return 100

    def __getitem__(self, idx):
        return self.data[idx]
