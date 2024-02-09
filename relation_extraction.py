"""replicate GPT-RE"""
import json
import statistics
import os
import pandas as pd
import argparse
import math
from tqdm import tqdm
from transformers import pipeline
from LLM_API import LLMApi
import random
import numpy as np
from testeval import compute_f1
from shared.const import semeval_reltoid
from shared.const import semeval_idtoprompt
from shared.const import ace05_reltoid
from shared.const import ace05_idtoprompt
from shared.const import tacred_reltoid
from shared.const import scierc_reltoid
from shared.const import wiki_reltoid
from shared.prompt import instance
from sklearn.metrics import classification_report
from knn_simcse import find_knn_example, find_lmknn_example
from simcse import SimCSE

from shared.prompt import generate_zero_prompt
from shared.prompt import generate_select_prompt
from shared.prompt import generate_select_auto_prompt
from shared.result import get_results_onebyone
from shared.result import get_results_select

           #print(prompt_query)

def generate_relation_dict_label(dataset):
    labels = []
    with open(dataset, "r") as f:
        relation_dict = {}
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)

            
            if tmp_dict["relations"]== [[]]:
                rel = "None"
            else:
                rel = tmp_dict["relations"][0][0][4]
            if rel not in relation_dict.keys():
                relation_dict[rel] = len(relation_dict.keys())
            
            labels.append(relation_dict[rel])

        print(relation_dict)
        return relation_dict, labels
def generate_label(dataset, relation_dict):
    labels = []
    with open(dataset, "r") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)

            
            if tmp_dict["relations"]== [[]]:
                rel = "NONE"
            else:
                rel = tmp_dict["relations"][0][0][4]
            if rel not in relation_dict.keys():
                relation_dict[rel] = len(relation_dict.keys())
            
            labels.append(relation_dict[rel])

        print(relation_dict)
        return labels



def generate_query(h_type, t_type, relation_list, query_dict):
    query_list = []
    #print(query_dict)
    for rel in relation_list:
        if rel == "None":
            continue
        else:
            query = query_dict[str((h_type,rel,t_type))]
            query_list.append(query)
    return query_list



def build_query_dict(dataset):
    with open("query_templates/ace2005.json", "r") as f:
        whole_dict = json.load(f)
        query_dict = whole_dict["qa_turn2"]
        return query_dict



def get_train_example(example_path, reltoid, no_na):
    example_dict = {k:list() for k in reltoid.values()}
    with open(example_path, "r") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            if tmp_dict["relations"] == [[]]:
                rel = "NONE"
                example_dict[reltoid[rel]].append(tmp_dict)
            else:
                rel = tmp_dict["relations"][0][0][4]
                example_dict[reltoid[rel]].append(tmp_dict)
    return example_dict
     
def get_test_example(example_path, reltoid):
    example_dict = {k:list() for k in reltoid.values()}
    with open(example_path, "r") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            if tmp_dict["relations"] == [[]]:
                rel = "NONE"
                example_dict[reltoid[rel]].append(tmp_dict)
            else:
                rel = tmp_dict["relations"][0][0][4]
                example_dict[reltoid[rel]].append(tmp_dict)
    return example_dict
     

def auto_generate_example(example_dict, reltoid, idtoprompt, num_per_rel, num_na, random_label, reasoning, llm):
    #ratio = 0.5
    #num_per_rel = 4
    num_example = num_per_rel * (len(example_dict.keys()) - 1) + num_na


    #select_dict = {"0":0, "A":1,"B":2,"C":3,"D":4,"E":5,"F":6}
    #reltoalpha = {0:"0", 1:"A", 2:"B", 3:"C", 4:"D", 5:"E", 6:"F"}
    #reltoalpha = {0:"NONE", 1:"Physical", 2:"General and affiliation", 3:"Person and social", 4:"Organization and affiliation", 5:"Part and whole", 6:"Agent and artifact"}
    #reltoalpha = {0:"NONE", 1:"PHYSICAL", 2:"GENERAL AND AFFILIATION", 3:"PERSON AND SOCIAL", 4:"ORGANIZATION AND AFFILIATION", 5:"PART AND WHOLE", 6:"AGENT AND ARTIFACT"}
           #else:
            #    if random.random() > 0.9:
            #        example_list.append(tmp_dict)
            #    else:
            #        continue
    #examples = [item for k,v in example_dict.items() for item in v]
    examples = []
    for relid in example_dict.keys():
        if relid == 0:
            examples.append(random.sample(example_dict[relid], num_na))
        else:
            examples.append(random.sample(example_dict[relid], num_per_rel))
            

    flat_examples = [item for sublist in examples for item in sublist]
    #print(len(examples))
    example_list = random.sample(flat_examples, num_example)
    #assert False
    
    example_prompt = str()
    for tmp_dict in example_list:
        string = " ".join(tmp_dict["sentences"][0])
        sub_head = tmp_dict["ner"][0][0][0]
        sub_tail = tmp_dict["ner"][0][0][1] + 1

                
        obj_head = tmp_dict["ner"][0][1][0]
        obj_tail = tmp_dict["ner"][0][1][1] + 1

        entity1 = " ".join(tmp_dict["sentences"][0][sub_head:sub_tail])
        entity1_type = tmp_dict["ner"][0][0][2]
        entity2 = " ".join(tmp_dict["sentences"][0][obj_head:obj_tail])
        entity2_type = tmp_dict["ner"][0][1][2]

        if random_label:
            rel = random.choice([x for x in reltoid.keys()])
        elif tmp_dict["relations"] == [[]]:
            rel = 'NONE'
        else:
            rel = tmp_dict["relations"][0][0][4]


        if not reasoning:
            prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ".\n"
        else:
            #tmp_query = "\nGiven the sentence: \"" + string + "\", What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + "?"
            tmp_query = "What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + " in the sentence \"" + string + "\"?"
            prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ". It is because:"
            #print(prompt_query)
            #assert False

            results = llm.process_inputs(tmp_query)
            prompt_query = prompt_query + results[0] +"\n"
            #print(prompt_query)
            #assert False
        example_prompt += prompt_query
    return example_prompt



def find_prob(target, result, probs):
    try:
        index = [x.strip() for x in probs["tokens"]].index(str(target))
        #print(probs["token_logprobs"][index])
        return math.exp(probs["token_logprobs"][index])
    except:
        len_target = len(target)
        for i in range(2, len_target+1):
            for j in range(len(probs["tokens"])):
                if i + j > len(probs["tokens"]):
                    continue
                #print(j+i)
                #print(len(probs["tokens"]))
                tmp_word = "".join([probs["tokens"][x] for x in range(j, j+i)])
                if tmp_word.strip() != target:
                    #print(tmp_word.strip())
                    continue
                else:
                    #print(tmp_word.strip())

                    start = j
                    end = j + i
                    sum_prob = 0
                    for k in range(start, end):
                        sum_prob += math.exp(probs["token_logprobs"][k])
                        #print(sum_prob)
                    #if sum_prob == None:
                        #print(target)
                        #print(result)
                    return sum_prob / i
        return 0.0
def smooth(x):
    if True:
        return np.exp(x)/sum(np.exp(x)) 
    else:
        return x


    
def compute_variance(knn_distribution):
    count_dis = [0 for x in range(len(knn_distribution))]
    for i in knn_distribution:
        count_dis[i] += 1
    tmp_distribution = 1.0 * np.array(count_dis)
    
    var = statistics.variance(tmp_distribution)
    print(var)
    if np.argmax(tmp_distribution) == 0 and var < 5:
        return 1
    else:
        return 0

def generate_ft_example(tmp_dict, ft_dict, reltoid, idtoprompt, demo, args):
    tmp_example = instance(tmp_dict)

    example_list = ft_dict[tmp_example.id]
    if args.reverse:
        example_list.reverse()
    label_other = 0
    tmp_knn = []
    example_prompt = str()
    if args.var:
        knn_distribution = []
        for tmp_dict in example_list:
            if tmp_dict["relations"] == [[]]:
                rel = 'NONE'
            else:
                rel = tmp_dict["relations"][0][0][4]
            knn_distribution.append(reltoid[rel])
        label_other = compute_variance(knn_distribution)
    for tmp_dict in example_list:
        string = " ".join(tmp_dict["sentences"][0])
        sub_head = tmp_dict["ner"][0][0][0]
        sub_tail = tmp_dict["ner"][0][0][1] + 1

                
        obj_head = tmp_dict["ner"][0][1][0]
        obj_tail = tmp_dict["ner"][0][1][1] + 1

        entity1 = " ".join(tmp_dict["sentences"][0][sub_head:sub_tail])
        entity1_type = tmp_dict["ner"][0][0][2]
        entity2 = " ".join(tmp_dict["sentences"][0][obj_head:obj_tail])
        entity2_type = tmp_dict["ner"][0][1][2]

        if args.random_label:
            rel = random.choice([x for x in reltoid.keys()])
        elif tmp_dict["relations"] == [[]]:
            rel = 'NONE'
        else:
            rel = tmp_dict["relations"][0][0][4]
        tmp_knn.append(reltoid[rel])

        tmp_example = instance(tmp_dict)
        if not args.reasoning or label_other == 1:
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ".\n"
            #prompt_query = instance(tmp_dict).reference + " is " + idtoprompt[reltoid[rel]] + ".\n\n"
        elif args.self_error:
            prompt_query = tmp_example.get_self_error(tmp_dict, demo, reltoid, idtoprompt, args)
        else:
            #tmp_query = "\nGiven the sentence: \"" + string + "\", What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + "?"
            tmp_query = "What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + " in the sentence \"" + string + "\"?"
            #print(prompt_query)
            #assert False

            while(True):
                try:
                    results, probs = demo.get_multiple_sample(tmp_query)
                    break
                except:
                    continue
            #prompt_query = prompt_query + results[0] +"\n"
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.clue + results[0] + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ". It is because:\n" + results[0] + "\n"
            #print(prompt_query)
            #assert False
        example_prompt += prompt_query
    return example_prompt, tmp_knn, label_other, example_list



def generate_lm_example(gpu_index_flat, tmp_dict, train_dict, train_sentences, k, reltoid, idtoprompt, num_per_rel, num_na, random_label, reasoning, demo, var, args):
    #train_list = [x for y in train_dict.values() for x in y]
    #print(tmp_dict)
    #assert False
    #print(len(train_list))
    example_list = find_lmknn_example(gpu_index_flat, tmp_dict,train_dict,train_sentences, k)
    
    if args.reverse:
        example_list.reverse()
    label_other = 0
    tmp_knn = []
    example_prompt = str()
    if var:
        knn_distribution = []
        for tmp_dict in example_list:
            if tmp_dict["relations"] == [[]]:
                rel = 'NONE'
            else:
                rel = tmp_dict["relations"][0][0][4]
            knn_distribution.append(reltoid[rel])
        label_other = compute_variance(knn_distribution)
    for tmp_dict in example_list:
        string = " ".join(tmp_dict["sentences"][0])
        sub_head = tmp_dict["ner"][0][0][0]
        sub_tail = tmp_dict["ner"][0][0][1] + 1

                
        obj_head = tmp_dict["ner"][0][1][0]
        obj_tail = tmp_dict["ner"][0][1][1] + 1

        entity1 = " ".join(tmp_dict["sentences"][0][sub_head:sub_tail])
        entity1_type = tmp_dict["ner"][0][0][2]
        entity2 = " ".join(tmp_dict["sentences"][0][obj_head:obj_tail])
        entity2_type = tmp_dict["ner"][0][1][2]

        if random_label:
            rel = random.choice([x for x in reltoid.keys()])
        elif tmp_dict["relations"] == [[]]:
            rel = 'NONE'
        else:
            rel = tmp_dict["relations"][0][0][4]
        tmp_knn.append(reltoid[rel])

        tmp_example = instance(tmp_dict)
        if not reasoning or label_other == 1:
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ".\n"
            #prompt_query = instance(tmp_dict).reference + " is " + idtoprompt[reltoid[rel]] + ".\n\n"
        else:
            #tmp_query = "\nGiven the sentence: \"" + string + "\", What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + "?"
            tmp_query = "What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + " in the sentence \"" + string + "\"?"
            #print(prompt_query)
            #assert False

            while(True):
                try:
                    results, probs = demo.get_multiple_sample(tmp_query)
                    break
                except:
                    continue
            #prompt_query = prompt_query + results[0] +"\n"
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.clue + results[0] + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ". It is because:\n" + results[0] + "\n"
            #print(prompt_query)
            #assert False
        example_prompt += prompt_query
    return example_prompt, tmp_knn, label_other, example_list



def generate_knn_example(knn_model, tmp_dict, train_dict, k, reltoid, idtoprompt, num_per_rel, num_na, random_label, reasoning, demo, var, args):
    #train_list = [x for y in train_dict.values() for x in y]
    #print(tmp_dict)
    #assert False
    #print(len(train_list))
    example_list = find_knn_example(knn_model, tmp_dict,train_dict,k, args.entity_info)
    
    if args.reverse:
        example_list.reverse()
    label_other = 0
    tmp_knn = []
    example_prompt = str()
    if var:
        knn_distribution = []
        for tmp_dict in example_list:
            if tmp_dict["relations"] == [[]]:
                rel = 'NONE'
            else:
                rel = tmp_dict["relations"][0][0][4]
            knn_distribution.append(reltoid[rel])
        label_other = compute_variance(knn_distribution)
    for tmp_dict in example_list:
        string = " ".join(tmp_dict["sentences"][0])
        sub_head = tmp_dict["ner"][0][0][0]
        sub_tail = tmp_dict["ner"][0][0][1] + 1

                
        obj_head = tmp_dict["ner"][0][1][0]
        obj_tail = tmp_dict["ner"][0][1][1] + 1

        entity1 = " ".join(tmp_dict["sentences"][0][sub_head:sub_tail])
        entity1_type = tmp_dict["ner"][0][0][2]
        entity2 = " ".join(tmp_dict["sentences"][0][obj_head:obj_tail])
        entity2_type = tmp_dict["ner"][0][1][2]

        if random_label:
            rel = random.choice([x for x in reltoid.keys()])
        elif tmp_dict["relations"] == [[]]:
            rel = 'NONE'
        else:
            rel = tmp_dict["relations"][0][0][4]
        tmp_knn.append(reltoid[rel])

        tmp_example = instance(tmp_dict)
        if not reasoning or label_other == 1:
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ".\n"
            #prompt_query = instance(tmp_dict).reference + " is " + idtoprompt[reltoid[rel]] + ".\n\n"
        else:
            #tmp_query = "\nGiven the sentence: \"" + string + "\", What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + "?"
            tmp_query = "What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + " in the sentence \"" + string + "\"?"
            #print(prompt_query)
            #assert False

            while(True):
                try:
                    results = demo.get_multiple_sample(tmp_query)
                    break
                except:
                    continue
            #prompt_query = prompt_query + results[0] +"\n"
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.clue + results[0] + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ". It is because:\n" + results[0] + "\n"
            #print(prompt_query)
            #assert False
        example_prompt += prompt_query
    return example_prompt, tmp_knn, label_other, example_list



def generate_ft_dict(args):
    ft_dict = {}
    knn_dict = {}
    train_dict = {}
    if args.use_dev and args.store_error_reason:
        knn_path = "./knn_ids/knn_ids_{}_train_dev.txt".format(args.task)
    elif args.use_dev:
        knn_path = "./knn_ids/knn_ids_{}_dev.txt".format(args.task)
    else:
        knn_path = "./knn_ids/knn_ids_{}.txt".format(args.task)
    with open(knn_path, "r") as f:
        num_id = 0
        for line in f.read().splitlines():
            knn_num = line.split(" ")
            ft_dict[num_id] = knn_num[:args.k]
            num_id += 1

    with open(args.test_dataset, "r") as f:
        num_id = 0
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            knn_dict[tmp_dict["doc_key"]] = ft_dict[num_id]
            num_id += 1
    with open(args.example_dataset, "r") as f:
        num_id = 0
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            train_dict[num_id] = tmp_dict
            num_id += 1
    knn_ft_dict = {}
    for key in knn_dict.keys():
        #print(knn_dict[key])
        #print(train_dict)
        knn_ft_dict[key] = [train_dict[int(x)] for x in knn_dict[key]]
    return knn_ft_dict

def get_binary_select(pred, tmp_dict, llm, knn_list, reltoid, idtoprompt, args):
    test_example = instance(tmp_dict)
    prompt_list = str()
    for example in knn_list:
        knn_example = instance(example)
        if pred == reltoid[knn_example.rel]:
            prompt_list += knn_example.discriminator + idtoprompt[pred] + "?" + knn_example.answer + " yes.\n"
        else:

            prompt_list += knn_example.discriminator + idtoprompt[pred] + "?" + knn_example.answer + " no.\n"

    
    prompt_list += test_example.discriminator + idtoprompt[pred] + "?" + test_example.answer

    while True:
        try:
            results= llm.process_inputs(prompt_list)
            break
        except:
            continue
    
    #print(prompt_list)
    print(results[0])
    #assert False
    if "no" in results[0]:
        pred = 0
    return pred #, math.exp(probs[0]["token_logprobs"][0])


def run(reltoid, idtoprompt, store_path, args):
    
    llm = LLMApi()
    llm.load_model_and_config()

    example_dict = get_train_example(args.example_dataset, reltoid, args.no_na)
    test_dict = get_test_example(args.test_dataset, reltoid)
    flat_examples = [item for sublist in test_dict.values() for item in sublist]
    test_examples = random.sample(flat_examples, args.num_test)

    #train_list = test_examples
    train_list = [x for y in example_dict.values() for x in y]
    if args.entity_info:
        train_dict = {instance(x).reference:x for x in train_list}
        train_sentences = [instance(x).reference for x in train_list]
    else:
        train_dict = {instance(x).sentence:x for x in train_list}
        train_sentences = [instance(x).sentence for x in train_list]

    knn_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
    # for debugging beg
    
    train_sentences = train_sentences[:20]

    # for debugging end
    knn_model.build_index(train_sentences, device="cuda")

    print(len(test_examples))

    micro_f1 = 0.0
    for _ in range(args.num_run):
        if args.fixed_example:
            example_prompt = auto_generate_example(example_dict, reltoid, idtoprompt, args.num_per_rel, args.num_na, args.random_label, args.reasoning, llm)
            print(example_prompt)
        if not args.fixed_test:
            test_examples = random.sample(flat_examples, args.num_test)

        labels = []
        preds = []
        num = 0
        whole_knn = []
        azure_error = []
        for tmp_dict in test_examples:
            tmp_knn = []
            if tmp_dict["relations"] == [[]] and args.no_na:
                num += 1
                continue
 
            if tmp_dict["relations"] != [[]] and tmp_dict["relations"][0][0][4] == "Other" and args.no_na:
                num += 1
                continue
       
            if tmp_dict["relations"] != [[]] and tmp_dict["relations"][0][0][4] != "Other" and args.null:
                num += 1
                continue

            label_other = 0
            example_prompt, tmp_knn, label_other, knn_list = generate_knn_example(knn_model, tmp_dict, train_dict, args.k, reltoid, idtoprompt, args.num_per_rel, args.num_na, args.random_label, args.reasoning, llm, args.var, args)
            whole_knn.append(tmp_knn)
            num += 1
            if tmp_dict["relations"] == [[]]:
                labels.append(0)
            else:
                labels.append(reltoid[tmp_dict["relations"][0][0][4]])
            sentence = " ".join(tmp_dict["sentences"][0])

            prompt_list, subject, target = generate_select_auto_prompt(tmp_dict, example_prompt, reltoid, args.no_na, args.reasoning, args)
            
            pred ,error= get_results_select(llm, prompt_list, reltoid, idtoprompt, args.verbalize, args)
            if error:
                azure_error.append(tmp_dict["doc_key"])
            if args.discriminator and pred != 0:
                ori_pred = pred
                pred = get_binary_select(pred, tmp_dict, llm, knn_list, reltoid, idtoprompt, args)
                if pred != ori_pred:
                    print("work!")

            if args.task == "wiki80" and pred == 0:
                pred = labels[-1]
                
            preds.append(pred)
            f1_result = compute_f1(preds, labels)
            print(f1_result, end="\n")
            
            if preds[-1] != labels[-1]:
                
                with open("{}/negtive.txt".format(store_path), "w") as negf:
                    negf.write(prompt_list + "\n")
                    negf.write(str(reltoid) + "\n")
                    negf.write("Prediction: " + str(preds[-1]) + "\n")
                    negf.write("Gold: " + str(labels[-1]) + "\n")
                    negf.write(tmp_dict["doc_key"])
                    negf.write("\n-----------------\n")


            with open("{}/results.txt".format(store_path),"w") as negf:
                negf.write(prompt_list + "\n")
                    
                negf.write(str(reltoid) + "\n")
                negf.write("Prediction: " + str(preds[-1]) + "\n")
                negf.write("Gold: " + str(labels[-1]) + "\n")
                negf.write(str(f1_result))
                negf.write("\n")
                negf.write(tmp_dict["doc_key"])
                negf.write("\n-----------------\n")
           
            print("processing:", 100*num/len(test_examples), "%", end="\n")
        print(classification_report(labels, preds, digits=4))
        report = classification_report(labels, preds, digits=4,output_dict=True)
        micro_f1 += f1_result["f1"]
        with open("{}/azure_error.csv".format(store_path), "w") as f:
            f.write('\n'.join([str(azure_error)]))
        with open("{}/knn.csv".format(store_path), "w") as f:
            for line in whole_knn:
                f.write('\n'.join([str(line)]))
                f.write("\n")
        df = pd.DataFrame(report).transpose()
        df.to_csv("{}/result_per_rel.csv".format(store_path))
        print(azure_error)
    avg_f1 = micro_f1 / args.num_run
    print("AVG f1:", avg_f1)
    print(args)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, required=True, choices=["ace05","semeval","tacred","scierc","wiki80"])
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--example_dataset", type=str, default=None, required=True)
    parser.add_argument("--test_dataset", type=str, default=None, required=True)
    parser.add_argument("--fixed_example", type=int, default=1)
    parser.add_argument("--fixed_test", type=int,default=1)
    parser.add_argument("--num_per_rel", type=int, default=2)
    parser.add_argument("--num_na", type=int, default=0)
    parser.add_argument("--no_na", type=int, default=0)
    parser.add_argument("--num_run", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_label", type=int, default=0)
    parser.add_argument("--reasoning", type=int, default=0)
    parser.add_argument("--use_knn", type=int, default=0)
    parser.add_argument("--lm_mask", type=int, default=0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--bert_sim", type=int, default=1)
    parser.add_argument("--var", type=int, default=0)
    parser.add_argument("--reverse", type=int, default=0)
    parser.add_argument("--verbalize", type=int, default=0)
    parser.add_argument("--entity_info", type=int, default=0)
    parser.add_argument("--structure", type=int, default=0)
    parser.add_argument("--use_ft", type=int, default=0)
    parser.add_argument("--self_error", type=int, default=0)
    parser.add_argument("--use_dev", type=int, default=0)
    parser.add_argument("--store_error_reason", type=int, default=0)
    parser.add_argument("--discriminator", type=int, default=0)
    parser.add_argument("--name", type=str, default=0)
    parser.add_argument("--null", type=str, default=1)

    tacred_idtoprompt = {tacred_reltoid[k]:k.upper() for k in tacred_reltoid.keys()}
    scierc_idtoprompt = {scierc_reltoid[k]:k.upper() for k in scierc_reltoid.keys()}
    wiki_idtoprompt = {wiki_reltoid[k]:k.upper() for k in wiki_reltoid.keys()}

    args = parser.parse_args()
    if args.null == 1:
        args.null = True
    else:
        args.null = False
    if args.lm_mask == 1:
        args.lm_mask = True
    else:
        args.lm_mask = False
    if args.verbalize == 1:
        args.verbalize = True
    else:
        args.verbalize = False

    if args.entity_info == 1:
        args.entity_info = True
    else:
        args.entity_info = False
    if args.reverse == 1:
        args.reverse = True
    else:
        args.reverse = False
    if args.var and args.no_na:
        raise Exception("Sorry, if focus on no NA examples, please turn var into 0")
    if args.var:
        args.var = True
    else:
        args.var = False
    if args.fixed_example and args.use_knn:
        assert False
    if args.fixed_example == 1:
        args.fixed_example = True
    else:
        args.fixed_example = False

    if args.fixed_test == 1:
        args.fixed_test = True
    else:
        args.fixed_test = False

    if args.reasoning == 1:
        args.reasoning = True
    else:
        args.reasoning = False

    if args.no_na == 1:
        args.no_na = True
    else:
        args.no_na = False

    if args.random_label == 1:
        args.random_label = True
    else:
        args.random_label = False
    print(args)
    if args.no_na and args.num_na != 0:
        print(args.no_na)
        print(args.num_na)
        assert False
    store_path = "./results/knn_{}_results/test={}_knn={}_reverse={}_nona={}_var={}_{}_{}_seed={}_{}_randomlabel={}_fixedex={}_fixedtest={}_Reason={}_Verbalize={}_Entityinfo={}_structure={}_useft={}_selferror={}_usedev={}_discri={}_{}".format(args.task, args.num_test, args.k, args.reverse, args.no_na, args.var, args.num_per_rel,args.num_na,args.seed,args.model,str(args.random_label),str(args.fixed_example),str(args.fixed_test), str(args.reasoning), args.verbalize, args.entity_info,args.structure, args.use_ft, args.self_error, args.use_dev, args.discriminator, args.name)
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    
    #task = sys.argv[1]
    #test_num = int(sys.argv[2])
    #seed = sys.argv[3]
    random.seed(args.seed)
    if args.task == "semeval":
        #example_dataset = "./dataset/semeval_gpt/train.json"
        #dataset = "./dataset/semeval_gpt/test.json"
        run(semeval_reltoid,semeval_idtoprompt, store_path, args)
    elif args.task == "ace05":
        #example_dataset = "./dataset/ace05/test.json"
        #dataset = "./dataset/ace05/ace05_0.2/ace05_0.2_test.txt"
        run(ace05_reltoid,ace05_idtoprompt, store_path, args)
    elif args.task == "tacred":
        run(tacred_reltoid, tacred_idtoprompt, store_path, args)
    elif args.task == "scierc":
        run(scierc_reltoid, scierc_idtoprompt, store_path, args)
    elif args.task == "wiki80":
        
        run(wiki_reltoid, wiki_idtoprompt, store_path, args)