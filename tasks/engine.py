import json

from module.utils import get_language_model
from module.utils import get_dataset
from module.utils import PromptingStrategy, set_seed

class Engine:
    def __init__(self, args):
        self.args = vars(args)
        for key, value in self.args.items():
            setattr(self, key, value)
        self.dataset_tag = self.dataset
        self._get_dataset()
        set_seed(self.seed)
        if ".json" in self.example_indices:
            with open(self.example_indices, 'r') as f:
                self.example_indices = json.load(f)
        elif self.example_indices == "all":
            self.example_indices = list(range(len(self.dataset)))
        else:
            self.example_indices = self.example_indices.strip().split(",")
            self.example_indices = [int(idx) for idx in self.example_indices]
            
    def _get_dataset(self):
        self.dataset = get_dataset(self.dataset, self.dataset_path)
            
    def _get_model(self):
        self.model = get_language_model(
            self.model_tag, 
            max_tokens=self.model_max_tokens, 
            temperature=self.model_temperature,
            batch_size=self.model_batch_size,
            load_in_4bit=self.load_in_4bit,
            thinking=self.model_thinking
        )
    