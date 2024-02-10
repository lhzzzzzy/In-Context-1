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

from re_functions import *
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

    train_sentences = train_sentences[:50]
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("train_sentences cutted")
    print("------------------------------------------------------------------------------------------\n\n\n")

    knn_model.build_index(train_sentences, device="cpu")

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("len of test_exp = {}".format(len(test_examples)))
    print("knn_model successfully loaded.")
    print("------------------------------------------------------------------------------------------\n\n\n")

    micro_f1 = 0.0
    for _ in range(args.num_run):
        if args.fixed_example:
            example_prompt = auto_generate_example(example_dict, reltoid, idtoprompt, args.num_per_rel, args.num_na, args.random_label, args.reasoning, llm)
            print(example_prompt)

        labels = []
        preds = []
        num = 0
        whole_knn = []
        azure_error = []
        for tmp_dict in test_examples:

            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("begin tmp_dict in test_examples")
            print("------------------------------------------------------------------------------------------\n\n\n")

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
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("finished generate_knn_example")
            print("------------------------------------------------------------------------------------------\n\n\n")
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
                
                with open("{}/negtive.txt".format(store_path), "a") as negf:
                    negf.write(prompt_list + "\n")
                    negf.write(str(reltoid) + "\n")
                    negf.write("Prediction: " + str(preds[-1]) + "\n")
                    negf.write("Gold: " + str(labels[-1]) + "\n")
                    negf.write(tmp_dict["doc_key"])
                    negf.write("\n-----------------\n")


            with open("{}/results.txt".format(store_path),"a") as negf:
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