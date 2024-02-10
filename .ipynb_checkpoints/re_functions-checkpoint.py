"""useful function for relation_extraction.py"""
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

from shared.prompt import instance
from sklearn.metrics import classification_report
from knn_simcse import find_knn_example, find_lmknn_example
from simcse import SimCSE


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
    num_example = num_per_rel * (len(example_dict.keys()) - 1) + num_na



    examples = []
    for relid in example_dict.keys():
        if relid == 0:
            examples.append(random.sample(example_dict[relid], num_na))
        else:
            examples.append(random.sample(example_dict[relid], num_per_rel))
            

    flat_examples = [item for sublist in examples for item in sublist]
    example_list = random.sample(flat_examples, num_example)
    
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
            tmp_query = "What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + " in the sentence \"" + string + "\"?"
            prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ". It is because:"

            results = llm.process_inputs(tmp_query)
            prompt_query = prompt_query + results[0] +"\n"
        
        example_prompt += prompt_query
    return example_prompt



def find_prob(target, result, probs):
    try:
        index = [x.strip() for x in probs["tokens"]].index(str(target))
        return math.exp(probs["token_logprobs"][index])
    except:
        len_target = len(target)
        for i in range(2, len_target+1):
            for j in range(len(probs["tokens"])):
                if i + j > len(probs["tokens"]):
                    continue
             
                tmp_word = "".join([probs["tokens"][x] for x in range(j, j+i)])
                if tmp_word.strip() != target:
                    continue
                else:

                    start = j
                    end = j + i
                    sum_prob = 0
                    for k in range(start, end):
                        sum_prob += math.exp(probs["token_logprobs"][k])
                  
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
        elif args.self_error:
            prompt_query = tmp_example.get_self_error(tmp_dict, demo, reltoid, idtoprompt, args)
        else:
            tmp_query = "What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + " in the sentence \"" + string + "\"?"
        

            while(True):
                try:
                    results, probs = demo.get_multiple_sample(tmp_query)
                    break
                except:
                    continue
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.clue + results[0] + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ". It is because:\n" + results[0] + "\n"
     
        example_prompt += prompt_query
    return example_prompt, tmp_knn, label_other, example_list



def generate_lm_example(gpu_index_flat, tmp_dict, train_dict, train_sentences, k, reltoid, idtoprompt, num_per_rel, num_na, random_label, reasoning, demo, var, args):

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
        else:
            tmp_query = "What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + " in the sentence \"" + string + "\"?"

            while(True):
                try:
                    results, probs = demo.get_multiple_sample(tmp_query)
                    break
                except:
                    continue
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.clue + results[0] + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ". It is because:\n" + results[0] + "\n"
        example_prompt += prompt_query
    return example_prompt, tmp_knn, label_other, example_list



def generate_knn_example(knn_model, tmp_dict, train_dict, k, reltoid, idtoprompt, num_per_rel, num_na, random_label, reasoning, llm, var, args):
    
    example_list = find_knn_example(knn_model, tmp_dict,train_dict,k, args.entity_info)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("finish find_knn_example")
    print("------------------------------------------------------------------------------------------\n")
    
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
        else:
            tmp_query = "What are the clues that lead the relation between \"" + entity1 + "\" and \"" + entity2 + "\" to be " + idtoprompt[reltoid[rel]] + " in the sentence \"" + string + "\"?"
            
            while(True):
                try:
                    # remeber the input is a list
                    results = llm.process_inputs([tmp_query])
                    break
                except:
                    continue
            if args.structure:
                prompt_query = tmp_example.prompt + tmp_example.clue + results[0] + tmp_example.pred + idtoprompt[reltoid[rel]] + "\n"
            else:
                prompt_query = "\nContext: " + string + "\n" + "Given the context, the relation between " + entity1 + " and " + entity2 + " is " + idtoprompt[reltoid[rel]] + ". It is because:\n" + results[0] + "\n"
            
            print("\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("reasoning done:")
            print(prompt_query)
            print("-----------------------------------------------------------------------------------\n\n\n")

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
    
    print(results[0])
    if "no" in results[0]:
        pred = 0
    return pred #, math.exp(probs[0]["token_logprobs"][0])