import json
import re
import time
from tqdm import tqdm
import spacy
from scispacy.linking import EntityLinker
from openie import StanfordOpenIE
import requests

from nltk.tokenize import sent_tokenize

from medcat.cat import CAT
cat = CAT.load_model_pack("umls_self_train_model_pt2ch_3760d588371755d0.zip")

def gen_entity_mask(doc):
    terms = cat.get_entities(doc)["entities"]
    mask_sents = []
    for value in terms.values():
        if value['acc'] >= 0.5:
            doc_list = list(doc)
            masked_doc_list = []
            masked_doc_list.extend(doc_list[:value['start']])
            masked_doc_list.append("[MASK]")
            masked_doc_list.extend(doc_list[value['end']:])
            masked_doc = ''.join(masked_doc_list)
            masked_doc_sentlist = sent_tokenize(masked_doc)
            ctx_list = []
            for sent in masked_doc_sentlist:
                if "[MASK]" in sent:
                    masked_sent = sent
                else:
                    ctx_list.append(sent)
            mask_sents.append({
                "source": " ".join(ctx_list),
                "target": value['source_value'],
                "masked_sent": masked_sent
            })
    return mask_sents

def entity_linker(nlp, document):
    doc = nlp(document)
    linked_ent = []
    orig_ent = []
    linker = nlp.get_pipe("scispacy_linker")
    for entity in doc.ents:
        for linker_ent in entity._.kb_ents:
            if linker_ent[1] > 0:
                linked_ent.append(linker.kb.cui_to_entity[linker_ent[0]].canonical_name)
                orig_ent.append(entity.text)
                break
    return orig_ent, linked_ent

# def gen_entity_mask(nlp, doc):
#     mask_sents = []
#     orig_ent, _ = entity_linker(nlp, doc)
#     for ent in orig_ent:
#         doc_sents = sent_tokenize(doc)
#         for sid, sent in enumerate(doc_sents):
#             if ent in sent:
#                 ctx_list = []
#                 ctx_list.extend(doc_sents[0:sid])
#                 ctx_list.extend(doc_sents[sid+1:])
#                 mask_sents.append({
#                     "source": " ".join(ctx_list),
#                     "target": ent,
#                     "masked_sent": sent.replace(ent, "[MASK]")
#                 })
#     return mask_sents

def ent_in_umls(ent):
    base_uri = 'https://uts-ws.nlm.nih.gov'
    path = '/search/current'
    query = {'string': ent, 'apiKey': '', 'rootSource': '',
             'termType': '', 'searchType': ''}
    output = requests.get(base_uri + path, params=query)
    output.encoding = 'utf-8'
    outputJson = output.json()
    results = (([outputJson['result']])[0])['results']
    if len(results) > 0:
        return True
    else:
        return False




def gen_re_mask(nlp, doc):
    sents = sent_tokenize(doc)
    mask_sents = []
    for sid, sent in enumerate(sents):
        ctx_list = []
        ctx_list.extend(sents[0:sid])
        ctx_list.extend(sents[sid+1:])
        ctx = " ".join(ctx_list)
        # 1. filter triples: sub / obj must belongs to medical entity
        #    (can link to scispacy[umls])
        triple_list = []
        with StanfordOpenIE() as client:
            try:
                triplets = client.annotate(sent)
            except Exception as err:
                time.sleep(10)
                print(err)
                triplets = client.annotate(sent)
            for triple in triplets:
                if ent_in_umls(triple['subject']) or ent_in_umls(triple['object']):
                    triple_list.append(triple)
        # print(triple_list)
        # 2. turn the triple list to an entity&relation list
        #    delete the duplicate entity/relation
        er_list = set()
        for t in triple_list:
            er_list.add(t['subject'])
            er_list.add(t['relation'])
            er_list.add(t['object'])
        # print(er_list)
        # 3. get the masked sentence
        #    mask the elements in er_list in turn
        for ele in er_list:
            ele_token_list = [tok.text for tok in nlp.tokenizer(ele)]
            ele_token_len = len(ele_token_list)
            sent_token_list = [tok.text for tok in nlp.tokenizer(sent)]
            sent_token_len = len(sent_token_list)
            for i in range(sent_token_len - ele_token_len + 1):
                if sent_token_list[i: i + ele_token_len] == ele_token_list:
                    mask_target = []
                    mask_target.extend(sent_token_list[:i])
                    mask_target.append('[MASK]')
                    mask_target.extend(sent_token_list[i+ele_token_len:])
                    masked_sent = " ".join(mask_target)
                    mask_sents.append({
                        "source": ctx,
                        "target": ele,
                        "masked_sent": masked_sent
                    })
    return mask_sents

def gen_num_mask(doc):
    mask_sents = []
    sents = sent_tokenize(doc)
    for sid, sent in enumerate(sents):
        ctx_list = []
        for cid, cent in enumerate(sents):
            if cid == sid:
                continue
            else:
                ctx_list.append(cent)
        ctx = " ".join(ctx_list)
        re_number = re.finditer(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', sent)
        for match in re_number:
            masked_sent = sent
            start, end = match.start(), match.end()
            target = masked_sent[start:end]
            masked_sent = masked_sent[:start] + '[MASK]' + masked_sent[end:]
            mask_sents.append({
                "source": ctx,
                "target": target,
                "masked_sent": masked_sent
            })
    return mask_sents

MASK_ENTITY = True
MASK_RE = True
MASK_NUM = True

if __name__ == '__main__':
    nlp = spacy.load("en_core_sci_scibert")
    nlp.add_pipe("scispacy_linker", config={"linker_name": "umls", "resolve_abbreviations": True})
    train_RRS = []
    with open('data.jsonl', encoding='utf-8') as rf:
        for line in rf:
            train_RRS.append(json.loads(line)["source"])
    mask_sents = []
    if MASK_ENTITY:
        for doc in tqdm(train_RRS, desc="Processing"):
            mask_sents = gen_entity_mask(doc)
            for mask_sent in mask_sents:
                with open("RRS_datasets/infill_entity_train.jsonl", 'a+', encoding='utf-8') as wf:
                    json.dump(mask_sent, wf, ensure_ascii=False)
                    wf.write("\n")
    if MASK_RE:
        for doc in tqdm(train_RRS, desc="Processing"):
            re_mask_sents = gen_re_mask(nlp, doc)
            for sent in re_mask_sents:
                with open("RRS_datasets/infill_re_train.jsonl", 'a+', encoding='utf-8') as wf:
                    json.dump(sent, wf, ensure_ascii=False, indent=4)
    if MASK_NUM:
        for doc in train_RRS:
            mask_sents.extend(gen_num_mask(doc))
        with open("RRS_datasets/infill_num_train.json", 'a+', encoding='utf-8') as wf:
            json.dump(mask_sents, wf, ensure_ascii=False, indent=4)
    
