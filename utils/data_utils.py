import copy
import json
import os
import re
from random import seed, choice, sample
from copy import deepcopy

import nltk.tokenize


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        dataset = []
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def read_corr_json(path):
    with open(path, encoding='utf-8') as rf:
        return json.load(rf)

def write_json(file_name: str, dataset):
    with open(file_name, 'w', encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)


def write_pred_data(args, text_path, predictions):
    dec_dir_name = os.path.join("./output", args.dataset + "_" + args.model_name, args.model_type, "dec_dir")
    ref_dir_name = os.path.join("./output", args.dataset + "_" + args.model_name, args.model_type, "ref_dir")

    json_fn = os.path.join("./output", args.dataset + "_" + args.model_name, args.model_type, args.ckpt_path.split('/')[-1].replace("ckpt", "json"))

    if not os.path.exists(dec_dir_name):
        os.makedirs(dec_dir_name)
    if not os.path.exists(ref_dir_name):
        os.makedirs(ref_dir_name)

    test_set = load_data(args, text_path)[-1]

    res = []
    sid = 0
    for data, pred in zip(test_set, predictions):
        with open(ref_dir_name + "/" + str(sid) + "_reference.txt", "w") as text_file:
            text_file.write("%s" % data["summary"])
        with open(dec_dir_name + "/" + str(sid) + "_decoded.txt", "w") as text_file:
            text_file.write("%s" % pred)
        res.append({
            "source_id": sid,
            "source_article_sentences": data["source"],
            "generated_summary_sentences": pred,
            "original_summary_sentences": data["summary"]
        })
        sid += 1
    with open(json_fn, "w", encoding='utf-8') as wf:
        wf.write(json.dumps(res, ensure_ascii=False, indent=4))

def write_corr_pred_data(args, predictions):
    dec_dir_name = os.path.join("./output", args.dataset + "_" + args.corr_model_name, args.model_type, "corr_instruct", args.corr_model_type, "dec_dir")
    ref_dir_name = os.path.join("./output", args.dataset + "_" + args.corr_model_name, args.model_type, "corr_instruct", args.corr_model_type,  "ref_dir")

    json_fn = os.path.join("./output", args.dataset + "_" + args.corr_model_name, args.model_type, "corr_instruct", args.corr_model_type, args.ckpt_path.split('/')[-1].replace("ckpt", "json"))

    if not os.path.exists(dec_dir_name):
        os.makedirs(dec_dir_name)
    if not os.path.exists(ref_dir_name):
        os.makedirs(ref_dir_name)

    test_set = load_data(args, args.test_fp)[-1]

    res = []
    sid = 0
    for data, pred in zip(test_set, predictions):
        with open(ref_dir_name + "/" + str(sid) + "_reference.txt", "w") as text_file:
            text_file.write("%s" % data["original_summary_sentences"])
        with open(dec_dir_name + "/" + str(sid) + "_decoded.txt", "w") as text_file:
            text_file.write("%s" % pred)
        res.append({
            "source_id": sid,
            "source_article_sentences": data["source_article_sentences"],
            "generated_summary_sentences": pred,
            "original_summary_sentences": data["original_summary_sentences"]
        })
        sid += 1
    with open(json_fn, "w", encoding='utf-8') as wf:
        wf.write(json.dumps(res, ensure_ascii=False, indent=4))

def load_data(args, text_path):
    train_set, val_set, test_set = [], [], []

    if args.do_train:
        train_set = read_json(os.path.join(text_path, "train.jsonl"))
        val_set = read_json(os.path.join(text_path, "val.jsonl"))
    elif args.do_test and (args.task == "CorrInstruction" or args.task == "Corr"):
        test_set = read_corr_json(args.test_fp)
    else:
        test_set = read_json(os.path.join(text_path, "test.jsonl"))

    if args.do_train:
        print("train_set:", train_set[0])
        print("val_set:", val_set[0])
    if args.do_test:
        print("test_set:", test_set[0])

    print("train", len(train_set), "val", len(val_set), "test", len(test_set))

    return train_set, val_set, test_set
