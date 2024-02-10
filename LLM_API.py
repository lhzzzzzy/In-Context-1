from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List
import deepspeed


class LLMApi:
    def __init__(self, model_name = "wxjiao/alpaca-7b"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.generation_config = None
    
    def load_model_and_config(self, temperature = 1.0, max_new_tokens = 200, do_sample = False, num_beams = 2, no_repeat_ngram_size = 2):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map = "auto") # I can use the model in multiple GPUs
            for param in self.model.parameters():
                print(param.device)
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


if __name__ == "__main__":
    model = LLMApi()
    model.load_model_and_config()
    # this model don't have padding tokens. can't run below's code.
    print(model.process_inputs(["hi, my name is hz liang, what's your name?", "which city is the capital of China? (a) Beijing (b) bilibili (c) youtube"]))
    