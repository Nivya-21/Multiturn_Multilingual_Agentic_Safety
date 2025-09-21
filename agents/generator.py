# agents/generator.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

class LlamaGenerator:
    """Manages the local Hugging Face model for generating responses."""
    def __init__(self, model_name: str, auth_token: str = None):
        try:
            if auth_token:
                login(auth_token)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"Generator model '{model_name}' loaded successfully.")
        except Exception as e:
            self.model = None
            self.tokenizer = None
            print(f"FATAL: Failed to load generator model '{model_name}': {e}")

    def generate_response(self, conversation: list, max_new_tokens: int = 2048) -> str:
        if not self.model or not self.tokenizer:
            return "Generator model not initialized."
        
        try:
            tokenized_chat = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            outputs = self.model.generate(tokenized_chat, max_new_tokens=max_new_tokens)
            return self.tokenizer.decode(outputs[0, tokenized_chat.shape[-1]:], skip_special_tokens=True)
        except Exception as e:
            return f"Error during response generation: {e}"