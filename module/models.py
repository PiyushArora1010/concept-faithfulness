from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Model:
    def __init__(
        self,
        name,
        max_tokens=256,
        temperature=0.7,
        padding_side="left",
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

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
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Repeat the same prompt n_completions times
        model_inputs = self.tokenizer(
            [text] * n_completions,
            return_tensors="pt",
            padding=True,
            # truncation=True,
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            num_return_sequences=n_completions,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        completions = []
        for i in range(n_completions):
            input_len = model_inputs.input_ids[i].shape[0]
            output_ids = generated_ids[i][input_len:]
            text_out = self.tokenizer.decode(
                output_ids, skip_special_tokens=True
            ).strip()
            completions.append(text_out)

        return completions

    def batch_generate_response(self, prompts):
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        completions = []
        for i in range(len(prompts)):
            input_len = model_inputs.input_ids[i].shape[0]
            output_ids = generated_ids[i][input_len:]
            text_out = self.tokenizer.decode(
                output_ids, skip_special_tokens=True
            ).strip()
            completions.append(text_out)

        return completions


class Qwen(Model):
    def __init__(
        self,
        name="Qwen/Qwen-2.5B-Chat",
        max_tokens=256,
        temperature=0.7,
        padding_side="left",
        thinking=True,
    ):
        super().__init__(
            name=name,
            max_tokens=max_tokens,
            temperature=temperature,
            padding_side=padding_side,
        )
        self.thinking = thinking
        
    def generate_response(self, prompt, n_completions=1):
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

        # Repeat the same prompt n_completions times
        model_inputs = self.tokenizer(
            [text] * n_completions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            num_return_sequences=n_completions,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        completions = []
        if not self.thinking:
            for i in range(n_completions):
                input_len = model_inputs.input_ids[i].shape[0]
                output_ids = generated_ids[i][input_len:]
                text_out = self.tokenizer.decode(
                    output_ids, skip_special_tokens=True
                ).strip()
                completions.append(text_out)
        else:            
            for i in range(n_completions):
                output_ids = generated_ids[i][len(model_inputs.input_ids[i]):].tolist()
                try:
                    # rindex finding 151668 (</think>)
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0
                text_out = self.tokenizer.decode(
                    output_ids[index:], skip_special_tokens=True
                ).strip()
                completions.append(text_out)
        
        return completions
        

    def batch_generate_response(self, prompts):
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.thinking,
            )
            for prompt in prompts
        ]

        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        completions = []
        if not self.thinking:
            for i in range(len(prompts)):
                input_len = model_inputs.input_ids[i].shape[0]
                output_ids = generated_ids[i][input_len:]
                text_out = self.tokenizer.decode(
                    output_ids, skip_special_tokens=True
                ).strip()
                completions.append(text_out)
        else:            
            for i in range(len(prompts)):
                output_ids = generated_ids[i][len(model_inputs.input_ids[i]):].tolist()
                try:
                    # rindex finding 151668 (</think>)
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0
                text_out = self.tokenizer.decode(
                    output_ids[index:], skip_special_tokens=True
                ).strip()
                completions.append(text_out)

        return completions

if __name__ == "__main__":
    import time
    
    model = Model(name="meta-llama/Llama-3.2-1B-Instruct", max_tokens=256, temperature=0.7, padding_side=None)
    prompt = "What is the capital of France?"
    
    timeStart = time.time()
    responses = model.generate_response(prompt, n_completions=4)
    timeEnd = time.time()
    print(f"Time taken for batch of 4: {timeEnd - timeStart} seconds")