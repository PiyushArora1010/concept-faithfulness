from module.utils import get_language_model
from module.utils import get_dataset
from module.utils import PromptingStrategy

class Engine:
    def __init__(self, args):
        self.args = vars(args)
        for key, value in self.args.items():
            setattr(self, key, value)
            
        self._get_dataset()
        self._get_model()
        
        if self.example_indices == "all":
            self.example_indices = list(range(len(self.dataset)))
        else:
            self.example_indices = self.example_indices.strip().split(",")
            self.example_indices = [int(idx) for idx in self.example_indices]
            if len(self.example_indices) == 2:
                self.example_indices = [idx for idx in range(self.example_indices[0], self.example_indices[-1]+1)]
            else:
                self.example_indices = [self.example_indices[0], self.example_indices[0]]
            
    def _get_dataset(self):
        self.dataset = get_dataset(self.dataset, self.dataset_path)
            
    def _get_model(self):
        self.model = get_language_model(
            self.model_tag, 
            max_tokens=self.model_max_tokens, 
            temperature=self.model_temperature,
            batch_size=self.model_batch_size,
            thinking=self.model_thinking
        )
    