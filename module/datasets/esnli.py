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
        return (
            f'Context: {s1}\nQuestion: Can we infer that {s2}?\n'
            f'Answer Choices:\n(A) Yes\n(B) No\n(C) Maybe, this is neutral'
        )
    
    def _process(self, dataset: Dataset):
        processed = []

        for ex in dataset:
            correct = self._label_map(ex['gold_label'])
            if correct is None:
                continue

            s1, s2 = ex['Sentence1'], ex['Sentence2']

            if s2 is None or s1 is None:
                continue

            # filter short sentences
            if len(s1.split()) < 3 or len(s2.split()) < 3:
                continue
            
            question = self._format_question(s1, s2)
            # question = self._add_instruction(question)

            answer_map = {
                'A': 'Yes',
                'B': 'No',
                'C': 'Maybe, this is neutral'
            }

            answer = f"{correct}) {answer_map[correct]}"

            processed.append((question, answer))

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            "question": self.data[idx][0],
            "answer": self.data[idx][1]
        }
        return item
    
if __name__ == "__main__":
    file_path = "/Users/piyush/Code/NLP/concept-faithfulness/data/e-SNLI"
    dataset = ESNLI(filepath=file_path, split="train")
    test_dataset = ESNLI(filepath=file_path, split="test")
    breakpoint()
    print(dataset[0]["question"])