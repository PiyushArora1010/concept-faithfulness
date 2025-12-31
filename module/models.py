from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Model:
    def __init__(
        self,
        name,
        max_tokens=256,
        temperature=0.7,
        batch_size=4,
        padding_side="left",
        thinking=False,
    ):
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.thinking = thinking

        # Load tokenizer
        if padding_side is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                name,
                trust_remote_code=True,
                padding_side=padding_side,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                name,
                trust_remote_code=True,
            )

        # Required for decoder-only models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, prompt, n_completions=1):
        prompts = [prompt] * n_completions
        return self.batch_generate_response(prompts)

    def batch_generate_response(self, prompts):
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **({'enable_thinking': self.thinking} if "qwen" in self.name.lower() else {})
            )
            for prompt in prompts
        ]

        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        completions = []
        
        for idx in range(0, len(texts), self.batch_size):
            end_idx = min(idx + self.batch_size, len(texts))
            batch_inputs = {k: v[idx:end_idx].to(self.model.device) for k, v in model_inputs.items()}
            
            generated_ids_batch = self.model.generate(
                **batch_inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Process this batch
            for i in range(len(generated_ids_batch)):
                global_idx = idx + i
                input_len = model_inputs.input_ids[global_idx].shape[0]
                output_ids = generated_ids_batch[i][input_len:].cpu()
                text_out = self.tokenizer.decode(
                    output_ids, skip_special_tokens=True
                ).strip()
                completions.append(text_out)

            del batch_inputs, generated_ids_batch

        return completions
    
if __name__ == "__main__":
    import time
    
    model = Model(name="meta-llama/Llama-3.1-8B-Instruct", max_tokens=256, temperature=0, padding_side="left")
    prompts = [
        "Capital of India in one word?",
        "Capital of USA in one word?",
    ]

    responses = model.batch_generate_response(prompts)
    
    for i, response in enumerate(responses):
        print(f"Response {i+1}:\n{response}\n")