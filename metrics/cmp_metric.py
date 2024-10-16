import argparse

import pyrouge

from evaluate import load
from questeval.questeval_metric import QuestEval
from summac.model_summac import SummaCZS, SummaCConv

import json


import os
os.environ['CURL_CA_BUNDLE'] = ''
import spacy
nlp = spacy.load("en_core_sci_scibert")
THRESH_HOLD_CAT = 0

def cmp_rouge_score(dec_dir, ref_dir):
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)

def cmp_bert_score(summary_list, dec_summary_list):
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=dec_summary_list, references=summary_list, model_type="allenai/scibert_scivocab_cased", lang="en")
    final_list = results['f1']
    final_score = sum(final_list) / len(final_list)

    return {
        "evaluate_bertscore": final_score
    }

def cmp_questeval_score(source_list, dec_summary_list):
    questeval = QuestEval(no_cuda=False)
    score = questeval.corpus_questeval(hypothesis=dec_summary_list, sources=source_list)
    return score

def cmp_far_score(source_ents, source_linked, ref_ents, ref_linked, dec_list):
    """
    FaR is Faithfulness-adjusted Recall.
    FaR = C / B + C
    C is the intersection of source, ref and system output.
    B + C is the intersection of source and ref.

    according to the description in the article
    [Towards Clinical Encounter Summarization: Learning to Compose Discharge Summaries from Prior Notes]
    "we use the medical named entity recognition (NER) system in SciSpacy (Neumann et al., 2019).
    The SciSpacy NER matches any spans in the text which might be an entity in UMLS,c a large biomedical database,
    and transforms the text into a set of medical entities.
    The cardinalities of the sets and their overlaps can then be used to calculate the above measures."

    Args:
        source_list:
        dec_summary_list:
        summary_list:

    Returns:

    """
    FaR_score = 0
    total = 0
    total_score = []
    for i in range(len(dec_list)):
        dec = dec_list[i]
        # B_C_list contains the entities linked in umls database
        B_C_list = [ent for ent in source_linked[i] if ent in ref_linked[i]]
        B_C_list = list(set(B_C_list))
        B_C = len(B_C_list)

        if B_C > 0:
            B_C_ents = []
            for ent in B_C_list:
                source_idx = [index for (index, item) in enumerate(source_linked[i]) if item == ent]
                B_C_ents.extend([source_ents[i][idx] for idx in source_idx])
                ref_idx = [index for (index, item) in enumerate(ref_linked[i]) if item == ent]
                B_C_ents.extend([ref_ents[i][idx] for idx in ref_idx])
            B_C_ents = list(set(B_C_ents))
            B_C_ents = sorted(B_C_ents, key=lambda x: (-len(x), x))

            C = 0
            for ent in B_C_ents:
                if ent in dec:
                    C += 1
                    dec = dec.replace(ent, "")

            FaR_score += C / B_C
            total += 1
            total_score.append(C / B_C)
        else:
            total_score.append(0)
    print(total_score)
    print(FaR_score)
    print(total)
    print(FaR_score / total)
    print(FaR_score / 100)
    return {
        "FaR": FaR_score / total  # the implement of denominator is different from famesumm, we think the length of B_C equals to 0 cannot be considered according to the original article.
    }

def cmp_summac_score(source_list, dec_summary_list):
    model_zs = SummaCZS(granularity="document", model_name="vitc",
                        device="cuda")
    final_score = []
    for i in range(len(source_list)):
        score = model_zs.score([source_list[i]], [dec_summary_list[i]])
        final_score.append(score["scores"][0])
    return {
        "SummaC": sum(final_score) / len(source_list)
    }

def cmp_cf1_score(dec_summary_list, ref_ents, ref_linked, summary_list):
    with open("HQS/ALL_medical_term_file.txt", "r", encoding='utf-8') as rf:
        medical_term_collection = rf.readlines()
        medical_term_collection = json.loads(medical_term_collection[0])
        medical_term_collection = sorted(medical_term_collection, key=lambda x: (-len(x), x))
        # print(medical_term_collection)
        print(len(medical_term_collection))
    with open("HQS/ALL_medical_term_map.json", "r", encoding='utf-8') as rf:
        medical_term_map = json.load(rf)

    correct, ref_num, dec_num = 0, 0, 0
    for i in range(len(dec_summary_list)):
        sent_ref_ents = ref_ents[i]
        sent_ref_ents = list(set(sent_ref_ents))
        sent_ref_ents = sorted(sent_ref_ents, key=lambda x: (-len(x), x))
        ref_num += len(sent_ref_ents)

        sent_ref_linked = ref_linked[i]
        sent_ref_linked = list(set(sent_ref_linked))

        dec = dec_summary_list[i]
        ref = summary_list[i]
        dec_ents = []
        for ent in nlp(dec).ents:
            dec_ents.append(ent.text)
        dec_ents = list(set(dec_ents))

        dec_linked = []
        supplement_ent = []
        for ent in dec_ents:
            isflag = False
            for key in medical_term_map.keys():
                if ent in medical_term_map[key]:
                    dec_linked.append(key)
                    isflag = True
                    break
            if not isflag:
                # print("new ent: " + ent)
                if ent in ref:
                    supplement_ent.append(ent)

        dec_num += len(dec_linked) + len(supplement_ent)
        correct_list = [ent for ent in sent_ref_linked if ent in dec_linked]
        # print(supplement_ent)
        correct_list.extend(supplement_ent)
        correct += len(correct_list)

    print(correct)
    print(ref_num)
    print(dec_num)
    recall = correct / ref_num
    precision = correct / dec_num
    f1 = 2 * recall * precision / (recall + precision)

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_fp', type=str, help='The json file for metric eval')
    parser.add_argument('--dec_dir', type=str, help='The dec director for metric eval')
    parser.add_argument('--ref_dir', type=str, help='The ref director for metric eval')
    parser.add_argument('--test_linked', type=str, help='The test dataset with linked entity (umls)')
    args = parser.parse_args()

    pred_fp = args.pred_fp
    dec_dir = args.dec_dir
    ref_dir = args.ref_dir
    test_linked = args.test_linked

    with open(pred_fp, encoding='utf-8') as rf:
        json_dict = json.load(rf)
    source_list = []
    summary_list = []
    dec_summary_list = []
    for jd in json_dict:
        source_list.append(jd['source_article_sentences'].strip())
        summary_list.append(jd['original_summary_sentences'].strip())
        dec_summary_list.append(jd['generated_summary_sentences'].strip())

    assert len(source_list)==len(summary_list)==len(dec_summary_list)

    source_ents = []
    source_linked = []
    ref_ents = []
    ref_linked = []
    with open(test_linked, encoding='utf-8') as rf:
        for line in rf:
            source_ents.append(json.loads(line)['source_ents'])
            source_linked.append(json.loads(line)['source_linked'])
            ref_ents.append(json.loads(line)['summary_ents'])
            ref_linked.append(json.loads(line)['summary_linked'])

    res_fp = pred_fp.replace(pred_fp.split('/')[-1], "results.txt")
    with open(res_fp, "a+", encoding='utf-8') as wf:
        # 1. compute Rouge score

        print("====================Rouge Score====================")
        final_score = cmp_rouge_score(dec_dir, ref_dir)
        print(final_score)
        final_score_select = {}
        final_score_select['rouge_1_f_score'] = final_score['rouge_1_f_score']
        final_score_select['rouge_2_f_score'] = final_score['rouge_2_f_score']
        final_score_select['rouge_l_f_score'] = final_score['rouge_l_f_score']
        wf.write("====================Rouge Score====================\n")
        json.dump(final_score_select, wf, ensure_ascii=False, indent=4)
        wf.write("\n")

        # 2. compute BertScore

        print("====================Bert Score====================")
        final_score = cmp_bert_score(summary_list, dec_summary_list)
        print(final_score)
        wf.write("====================Bert Score====================\n")
        json.dump(final_score, wf, ensure_ascii=False, indent=4)
        wf.write("\n")

        # 3. compute QuestEval score

        print("====================QuestEval Score====================")
        final_score = cmp_questeval_score(source_list, dec_summary_list)
        print(final_score)
        wf.write("====================QuestEval Score====================\n")
        json.dump({"corpus_score": final_score["corpus_score"]}, wf, ensure_ascii=False, indent=4)
        wf.write("\n")

        # 4. compute SummaC

        print("====================SummaC Score====================")
        final_score = cmp_summac_score(source_list, dec_summary_list)
        print(final_score)
        wf.write("====================SummaC Score====================\n")
        json.dump(final_score, wf, ensure_ascii=False, indent=4)
        wf.write("\n")

        # 5. compute FaR

        print("====================FaR Score====================")
        final_score = cmp_far_score(source_ents, source_linked, ref_ents, ref_linked, dec_summary_list)
        wf.write("====================FaR Score====================\n")
        json.dump(final_score, wf, ensure_ascii=False, indent=4)
        wf.write("\n")

        # 6. compute C F1

        print("====================CF1 Score====================")
        final_score = cmp_cf1_score(dec_summary_list, ref_ents, ref_linked, summary_list)
        print(final_score)
        wf.write("====================CF1 Score====================\n")
        json.dump(final_score, wf, ensure_ascii=False, indent=4)
        wf.write("\n")
