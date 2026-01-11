import os
from tasks.engine import Engine

import torch
from unsloth import FastLanguageModel

class TrainEngine(Engine):
    def __init__(self, args):
        super().__init__(args)
        
    def _get_model(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_tag,
            max_seq_length=self.model_max_tokens,
            load_in_4bit=False,  # False for LoRA 16bit
            fast_inference=True,  # Enable vLLM fast inference
            max_lora_rank=self.lora_rank,
            gpu_memory_utilization=0.6,  # Reduce if out of memory
        )
        self.model = model
        self.tokenizer = tokenizer
        
        if self.lora:
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],  # Remove QKVO if out of memory
                lora_alpha=self.lora_rank,
                use_gradient_checkpointing="unsloth",  # Enable long context finetuning
            )
            