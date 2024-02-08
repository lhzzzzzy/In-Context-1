"""replicate GPT-RE"""
from LLM_API import LLMApi
import argparse


def run(reltoid, idtoprompt, store_path, args):
    # demo = Demo(
    #         engine=args.model,
    #         temperature=0,
    #         max_tokens=256,
    #         top_p=1,
    #         frequency_penalty=0,
    #         presence_penalty=0,
    #         best_of=1,
    #         logprobs=1,
    #         )
    #relation_dict = {'None': 0, 'PHYS': 1, 'GEN-AFF': 2, 'PER-SOC': 3, 'ORG-AFF': 4, 'PART-WHOLE': 5, 'ART': 6}relation_dict = {'Others': 0, 'PHYS': 1, 'GEN-AFF': 2, 'PER-SOC': 3, 'ORG-AFF': 4, 'PART-WHOLE': 5, 'ART': 6}
    #reltoid = {'NONE': 0, 'PHYS': 1, 'GEN-AFF': 2, 'PER-SOC': 3, 'ORG-AFF': 4, 'PART-WHOLE': 5, 'ART': 6}
    #idtoprompt = {0: "NONE", 1: "PHYSICAL", 2: "GENERAL AND AFFILIATION", 3: "PERSON AND SOCIAL", 4: "ORGANIZATION AND AFFILIATION", 5: "PART AND WHOLE", 6: "AGENT AND ARTIFACT"}
    #relation_dict = {'OTHERS': 0, 'PHYS': 1, 'GEN-AFF': 2, 'PER-SOC': 3, 'ORG-AFF': 4, 'PART-WHOLE': 5, 'ART': 6}
    #query_dict = build_query_dict(dataset)
    #all_labels = generate_label(dataset, reltoid)

    model = LLMApi()
    model.load_model_and_config()

    example_dict = get_train_example(args.example_dataset, reltoid, args.no_na)
    test_dict = get_test_example(args.test_dataset, reltoid)
    flat_examples = [item for sublist in test_dict.values() for item in sublist]
    test_examples = random.sample(flat_examples, args.num_test)

    #train_list = test_examples
    train_list = [x for y in example_dict.values() for x in y]
    if args.no_na:
        if args.task == "semeval":
            train_list = [x for x in train_list if reltoid[x["relations"][0][0][4]] != 0]
        else:

            train_list = [x for x in train_list if x["relations"] != [[]]]
    #train_dict = {"The relation between" + "\"" + x["ner"][0][0][2] + "\" and \"" + x["ner"][0][1][2] + "\" in the sentence \"" + " ".join(x["sentences"][0]) + "\"":x for x in train_list}
    if not args.lm_mask:
        if args.entity_info:
            train_dict = {instance(x).reference:x for x in train_list}
            train_sentences = [instance(x).reference for x in train_list]
        else:
            train_dict = {instance(x).sentence:x for x in train_list}
            train_sentences = [instance(x).sentence for x in train_list]

        knn_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
        #knn_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        knn_model.build_index(train_sentences, device="cpu")
    else:
        train_dict = {instance(x).lm_mask:x for x in train_list}
        train_sentences = [instance(x).lm_mask for x in train_list]

        res = faiss.StandardGpuResources()

        index_flat = faiss.IndexFlatL2(1024)
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

        extractor = pipeline(model="roberta-large", task="feature-extraction")
        embed_array = []
        for item in tqdm(train_sentences):

            result = extractor(item, return_tensors=True)

            embeds = result[0].detach().numpy().copy()
            embed_array.append(embeds[-3,:])

        embed_list = np.array(embed_array)
        gpu_index_flat.add(embed_list)

    print(len(test_examples))

    micro_f1 = 0.0
    #example_prompt = auto_generate_example(example_dataset, relation_dict, 18, True)
    for run in range(args.num_run):
        if args.fixed_example:
            example_prompt = auto_generate_example(example_dict, reltoid, idtoprompt, args.num_per_rel, args.num_na, args.random_label, args.reasoning, demo)
            print(example_prompt)
        if not args.fixed_test:
            test_examples = random.sample(flat_examples, args.num_test)
        labels = []
        preds = []
        num = 0
        whole_knn = []
        whole_prob = []
        whole_prob_on_rel = []
        store_error_reason = {}
        azure_error = []
        for tmp_dict in test_examples:
            tmp_knn = []
            #tmp_dict = json.loads(line)
            #na_filter = random.random()
            #rel_filter = random.random()
            if tmp_dict["relations"] == [[]] and args.no_na:
                num += 1
                continue
            #elif tmp_dict["relations"] == [[]] and na_filter < 0.95:
            #    num += 1
            #    continue
            if tmp_dict["relations"] != [[]] and tmp_dict["relations"][0][0][4] == "Other" and args.no_na:
                num += 1
                continue
            #if rel_filter < 0.5:
            #    lineid += 1
            #    continue
            #elif tmp_dict["relations"] == [[]] and na_filter < 0.95:
            #    num += 1
            #    continue
            if tmp_dict["relations"] != [[]] and tmp_dict["relations"][0][0][4] != "Other" and args.null:
                num += 1
                continue
            #example_dict = get_train_example(example_dataset, reltoid)
            label_other = 0
            if not args.fixed_example and not args.use_knn:
                example_prompt = auto_generate_example(example_dict, reltoid, idtoprompt, args.num_per_rel, args.num_na, args.random_label, args.reasoning, demo)
            if args.use_knn:
                if args.use_ft:
                    example_prompt, tmp_knn, label_other, knn_list = generate_ft_example(tmp_dict, ft_dict, reltoid, idtoprompt, demo, args)
                elif args.lm_mask:
                    example_prompt, tmp_knn, label_other, knn_list = generate_lm_example(gpu_index_flat, tmp_dict, train_dict, train_sentences, args.k, reltoid, idtoprompt, args.num_per_rel, args.num_na, args.random_label, args.reasoning, demo, args.var, args)
                else:
                    example_prompt, tmp_knn, label_other, knn_list = generate_knn_example(knn_model, tmp_dict, train_dict, args.k, reltoid, idtoprompt, args.num_per_rel, args.num_na, args.random_label, args.reasoning, demo, args.var, args)
                whole_knn.append(tmp_knn)
            num += 1
            if tmp_dict["relations"] == [[]]:
                labels.append(0)
            else:
                labels.append(reltoid[tmp_dict["relations"][0][0][4]])
            sentence = " ".join(tmp_dict["sentences"][0])
            #prompt_list, subject, target = generate_zero_prompt(tmp_dict, query_dict, relation_dict.keys())

            prompt_list, subject, target = generate_select_auto_prompt(tmp_dict, example_prompt, reltoid, args.no_na, args.reasoning, args)
            #results, probs = demo.get_multiple_sample(prompt_list)
            #pred, prob_on_rel = get_results_onebyone(demo, prompt_list, target)
            #print(prompt_list)
            #assert False
            if args.var and label_other == 1:
                pred = 0
                prob_on_rel = 0
                prob = {"NONE": 1}
            else:
                pred, prob_on_rel, prob, error = get_results_select(demo, prompt_list, reltoid, idtoprompt, args.verbalize, args)
                if error:
                    azure_error.append(tmp_dict["doc_key"])
                if args.discriminator and pred != 0:
                    ori_pred = pred
                    pred, prob = get_binary_select(pred, tmp_dict, demo, knn_list, reltoid, idtoprompt, args)
                    if pred != ori_pred:
                        print("work!")

                if args.task == "wiki80" and pred == 0:
                    pred = labels[-1]
                
                #print(prob_on_rel)
                #assert False
            whole_prob.append(prob)
            whole_prob_on_rel.append(prob_on_rel)
            preds.append(pred)
            f1_result = compute_f1(preds, labels)
            print(f1_result, end="\n")
            
            if preds[-1] != labels[-1]:
                if args.store_error_reason:
                    error_reason = instance(tmp_dict).get_error_reason(preds[-1], tmp_dict, example_prompt, demo, idtoprompt, reltoid, args)
                    store_error_reason[instance(tmp_dict).id] = error_reason
                with open("{}/negtive.txt".format(store_path), "a") as negf:
                
                    #negf.write(args)
                    #negf.write("\n")
                    negf.write(prompt_list + "\n")
                    
                    negf.write(str(reltoid) + "\n")
                    negf.write(str(prob_on_rel) + "\n")
                    negf.write("Prediction: " + str(preds[-1]) + "\n")
                    #negf.write(preds[num])
                    negf.write("Gold: " + str(labels[-1]) + "\n")
                    negf.write(tmp_dict["doc_key"])
                    negf.write("\n-----------------\n")
            else:

                if args.store_error_reason:
                    correct_reason = instance(tmp_dict).get_correct_reason(demo, idtoprompt, reltoid, args)
                    store_error_reason[instance(tmp_dict).id] = correct_reason

            with open("{}/results.txt".format(store_path),"a") as negf:
                #negf.write(args)
                #negf.write("\n")
                negf.write(prompt_list + "\n")
                    
                negf.write(str(reltoid) + "\n")
                negf.write(str(prob_on_rel) + "\n")
                negf.write("Prediction: " + str(preds[-1]) + "\n")
                #negf.write(preds[num])
                negf.write("Gold: " + str(labels[-1]) + "\n")
                #negf.write(str(classification_report(labels[:num], preds, digits=4)))
                negf.write(str(f1_result))
                negf.write("\n")
                #negf.write(labels[num])
                negf.write(tmp_dict["doc_key"])
                negf.write("\n-----------------\n")
            #print(results[0])
            #print(probs[0])
            #if num > 100:
            #    assert False
            print("processing:", 100*num/len(test_examples), "%", end="\n")
        print(classification_report(labels, preds, digits=4))
        report = classification_report(labels, preds, digits=4,output_dict=True)
        if args.store_error_reason:
            with open("stored_reason/{}_dev.txt".format(args.task), "w") as f:
                json.dump(store_error_reason, f)
        with open("{}/labels.csv".format(store_path), "w") as f:
            f.write('\n'.join([str(labels)]))
        with open("{}/preds.csv".format(store_path), "w") as f:
            f.write('\n'.join([str(preds)]))
        with open("{}/probs.csv".format(store_path), "w") as f:
            for prob in whole_prob:
                json.dump(prob, f)
                f.write("\n")
        with open("{}/prob_on_rel.csv".format(store_path), "w") as f:
            f.write('\n'.join([str(x) for x in whole_prob_on_rel]))
        micro_f1 += f1_result["f1"]
        with open("{}/azure_error.csv".format(store_path), "w") as f:
            f.write('\n'.join([str(azure_error)]))
        with open("{}/knn.csv".format(store_path), "w") as f:
            for line in whole_knn:
                f.write('\n'.join([str(line)]))
                f.write("\n")
        df = pd.DataFrame(report).transpose()
        df.to_csv("{}/result_per_rel.csv".format(store_path))
        #print(report)
        print(azure_error)
        #assert False
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