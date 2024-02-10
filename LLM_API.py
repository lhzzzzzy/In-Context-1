from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List
import torch
from torch.cuda.amp import autocast
import os

class LLMApi:
    def __init__(self, model_name = "wxjiao/alpaca-7b"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
    
    def load_model_and_config(self, temperature = 1.0, max_new_tokens = 100, do_sample = False, num_beams = 2, no_repeat_ngram_size = 2):
        try:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            torch.cuda.set_per_process_memory_fraction(0.98, 0)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
            # self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)  # Move model to CUDA if available
            self.model.half()  # Convert model to half precision
            self.generation_config = GenerationConfig(
                                        temperature = temperature,
                                        max_new_tokens = max_new_tokens,
                                        do_sample = do_sample,
                                        num_beams = num_beams,
                                        no_repeat_ngram_size = no_repeat_ngram_size,
                                    )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def process_inputs(self, prompts:List[str]):
        '''prompts should be a list of input'''
        model_inputs = self.tokenizer(prompts, return_tensors="pt")
        with autocast(): # half precision
            generated_ids = self.model.generate(** model_inputs, generation_config = self.generation_config)
        output = self.tokenizer.batch_decode(generated_ids,skip_special_tokens=True)
        
        for i in range(len(output)): # only return real outputs
            output[i] = output[i][len(prompts[i]):]
        return output

