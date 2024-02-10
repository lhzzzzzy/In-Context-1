from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List
import torch

class LLMApi:
    def __init__(self, model_name = "wxjiao/alpaca-7b"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("I use {}!".format(self.device))
        print("-------------------------------------------------------------\n\n\n")
    
    def load_model_and_config(self, temperature = 1.0, max_new_tokens = 200, do_sample = False, num_beams = 2, no_repeat_ngram_size = 2):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
            # self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)  # Move model to CUDA if available
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
        generated_ids = self.model.generate(** model_inputs, generation_config = self.generation_config)
        output = self.tokenizer.batch_decode(generated_ids,skip_special_tokens=True)
        for i in range(len(output)): # only return real outputs
            output[i] = output[i][len(prompts[i]):]
        return output

