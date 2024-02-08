from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig




tokenizer = AutoTokenizer.from_pretrained("wxjiao/alpaca-7b")
model = AutoModelForCausalLM.from_pretrained("wxjiao/alpaca-7b")

my_generation_config = GenerationConfig(
    tempreature = 1.0,
    max_length = 500,
    do_sample = False,
    num_beams = 2,
    no_repeat_ngram_size = 2,
)


print("=============================================================================")
task_definition = "I'm a knowledgeable person. I will solve the relation extraction (RE) task. Given the context, "+\
        "I'll output the most precise relation between two entities choosing from the following six possible relations.\n\n"+\
            "PHYSICAL: located, near\nGENERAL AND AFFILIATION: citizen, resident, religion, ethnicity, organization location\n"+\
                "PERSON AND SOCIAL: business,family,lasting personal\n"+\
                    "ORGANIZATION AND AFFILIATION: employment,founder,ownership,student alumn,sports affiliation,investor shareholder,membership\n"+\
                        "PART AND WHOLE: artifact,geographical,subsidiary\n"+\
                            "AGENT AND ARTIFACT: user, owner, inventor, manufacturer\n" + \
                                "Bisides, I will only output the answer without any other information. \n\n" 

icl_demo = "Context: Virginia is falling behind when we should be ahead with schools like Tech developing the technology." \
                    + "Given the context, the relation between Virginia and schools is: PART AND WHOLE" + "\n\n" \
           + "Context: Now, representatives of the committee -- the International Committee for the Red Cross, who have been to some of the hospitals in Baghdad, report higher-than-usual numbers of casualties coming into those hospitals." + \
        "Given the context, the relation between casualties and hospitals is: PHYSICAL\n\n"

test_input = "Context: In recent weeks, the U.S. military has been transporting military equipment from bases in Germany to the Gulf through the port of Antwerp." + \
"Given the context, the relation between military and U.S is:"
    
tot_prompt = task_definition + icl_demo + test_input


prompts = [tot_prompt]

model_inputs = tokenizer(prompts, return_tensors="pt")

generated_ids = model.generate(** model_inputs, generation_config = my_generation_config)

outputs = tokenizer.batch_decode(generated_ids,skip_special_tokens=True)

print(outputs[0])