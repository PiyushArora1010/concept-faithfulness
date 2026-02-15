import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class Model:
    def __init__(
        self,
        name,
        max_tokens=256,
        temperature=0.7,
        batch_size=64,
        load_in_4bit=False,
        padding_side="left",
        thinking=False,
    ):
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.thinking = thinking

        self.tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            padding_side=padding_side,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = LLM(
            model=name,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype="auto",
            # load_in_4bit=load_in_4bit,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            # download_dir="downloads",
        )

    def generate_response(self, prompt, n_completions=1):
        responses = self.batch_generate_response([prompt], n_completions=n_completions)
        return responses

    def batch_generate_response(self, prompts, n_completions=1):
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **(
                    {"enable_thinking": self.thinking}
                    if "qwen" in self.name.lower()
                    else {}
                ),
            )
            for prompt in prompts
        ]

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=n_completions,
        )

        all_outputs = []

        for idx in range(0, len(texts), self.batch_size):
            batch_texts = texts[idx : idx + self.batch_size]
            results = self.llm.generate(batch_texts, sampling_params)

            # results: List[RequestOutput]
            # each result.outputs has n_completions entries
            for r in results:
                completions = [
                    o.text.strip() for o in r.outputs
                ]
                all_outputs.extend(completions)

        return all_outputs

if __name__ == "__main__":
    import time

    model = Model(
        name="/rds/general/user/pa524/home/concept-faithfulness/results/SFT/bbq/Qwen3_4B_all_concepts/model",
        max_tokens=256,
        temperature=0.7,
        padding_side="left",
    )
    # prompts = [
    #     "Capital of India in one word?",
    #     "Capital of USA in one word?",
    # ]
    prompt = "Give a random name of a city in Germany."

    responses = model.generate_response(prompt, n_completions=3)

    for i, response in enumerate(responses):
        print(f"Response {i+1}:\n{response}\n")