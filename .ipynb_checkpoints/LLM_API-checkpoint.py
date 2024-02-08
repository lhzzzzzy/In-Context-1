from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List

class LLMApi:
    def __init__(self, model_name = "wxjiao/alpaca-7b"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        self.output = None
    
    def load_model_and_config(self, tempreature = 1.0, max_length = 1000, do_sample = False, num_beams = 2, no_repeat_ngram_size = 2):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.generation_config = GenerationConfig(
                                        tempreature = tempreature,
                                        max_length = max_length,
                                        do_sample = do_sample,
                                        num_beams = num_beams,
                                        no_repeat_ngram_size = no_repeat_ngram_size,
                                    )

    def process_inputs(self, prompts:List[str]):
        '''prompts should be a list of input'''
        model_inputs = self.tokenizer(prompts, return_tensors="pt")
        generated_ids = self.model.generate(** model_inputs, generation_config = self.generation_config)
        return self.tokenizer.batch_decode(generated_ids,skip_special_tokens=True)

