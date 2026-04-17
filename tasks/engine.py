from module.utils import get_language_model, get_dataset, set_seed

class Engine:
    def __init__(self, args):
        self.args = vars(args)
        for key, value in self.args.items():
            setattr(self, key, value)
        set_seed(self.seed)
        self.dataset_tag = self.dataset
        self._get_dataset()
            
    def _get_dataset(self):
        self.dataset = get_dataset(self.dataset, self.dataset_path, self.split, self.sample_size)
        
        # remove this later
        import json
        indices_file = "indices_train.json"
        with open(indices_file, 'w') as f:
            json.dump(self.dataset.indices, f)
            
    def _get_model(self):
        self.model = get_language_model(
            self.model_tag, 
            max_tokens=self.model_max_tokens, 
            temperature=self.model_temperature,
            batch_size=self.model_batch_size,
            load_in_4bit=self.load_in_4bit,
            thinking=self.model_thinking
        )
    