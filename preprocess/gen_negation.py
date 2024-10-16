import json

import pandas as pd
import spacy
from tqdm import tqdm
from rouge import Rouge

neg_unigrams = ["no", "nope", "doesn't", "don't", "not"]

def get_relevant_ids(doc_sents_list, generated_summary_sent):
    relevant_score = {}
    rouge = Rouge()
    for doc_id, doc_sent in enumerate(doc_sents_list):
        rouge_score = rouge.get_scores([doc_sent], [generated_summary_sent])
        # print(rouge_score)
        rouge_l = rouge_score[0]['rouge-l']['f']
        relevant_score[doc_id] = rouge_l
    print(relevant_score)
    sorted_dict = dict(sorted(relevant_score.items(), key=lambda item: item[1], reverse=True))
    if len(list(sorted_dict.keys())) >= 3:
        top_three_keys = list(sorted_dict.keys())[:3]
    else:
        top_three_keys = list(sorted_dict.keys())
    print(top_three_keys)
    return top_three_keys


if __name__ == '__main__':
    nlp = spacy.load("en_core_sci_sm")
    filename = '../datasets/HQS/train.xlsx'
    sheetname = 'QS'
    df = pd.read_excel(filename, sheetname)
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
        doc = row['CHQ']
        doc_sents = nlp(doc).sents
        doc_sents_list = [sent.text for sent in doc_sents]
        print(doc_sents_list)
        sum = row['Summary']
        sum_sents = nlp(sum).sents
        sum_sents_list = [sent.text for sent in sum_sents]
        print(sum_sents_list)
        for idx, sum_sent in enumerate(sum_sents_list):
            negation_detected = False
            sum_sent_tokens = [token.text for token in nlp(sum_sent)]
            for neg in neg_unigrams:
                if neg in sum_sent_tokens:
                    sum_sent_tokens_alter = []
                    for token in sum_sent_tokens:
                        if negation_detected or token != neg:
                            sum_sent_tokens_alter.append(token)
                        if token == neg:
                            negation_detected = True
                    generated_summary_sent = " ".join(sum_sent_tokens_alter)
                    relevant_ids = get_relevant_ids(doc_sents_list, generated_summary_sent)
                    generated_summary_sentences = []
                    generated_summary_sentences.extend(sum_sents_list[:idx])
                    generated_summary_sentences.append(generated_summary_sent)
                    generated_summary_sentences.extend(sum_sents_list[idx+1:])
                    res = {
                        "source_article_sentences": doc_sents_list,
                        "original_summary_sentences": sum_sents_list,
                        "generated_summary_sentences": generated_summary_sentences,
                        "incorrect_sent_idx": idx,
                        "relevant_article_sent_indices": relevant_ids,
                        "original_summary_sent": sum_sent,
                        "generated_summary_sent": generated_summary_sent,
                        "label": 0,
                        "error_type": "Negation",
                        "err_span": "",
                        "corr_span": neg
                    }
                    with open("negation_corr_train.jsonl", "a+", encoding='utf-8') as wf:
                        json.dump(res, wf, ensure_ascii=False)
                        wf.write("\n")
