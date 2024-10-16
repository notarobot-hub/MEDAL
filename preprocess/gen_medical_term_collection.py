
import spacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
from tqdm import tqdm
import json
import os

THRESHOLD = 0


def entity_linker(nlp, document):
    doc = nlp(document)
    linked_ent = []
    orig_ent = []
    linker = nlp.get_pipe("scispacy_linker")
    for entity in doc.ents:
        for linker_ent in entity._.kb_ents:
            if linker_ent[1] > THRESHOLD:
                linked_ent.append(linker.kb.cui_to_entity[linker_ent[0]].canonical_name)
                orig_ent.append(entity.text)
                break
    return orig_ent, linked_ent


if __name__ == '__main__':
    nlp = spacy.load("en_core_sci_scibert")
    nlp.add_pipe("abbreviation_detector")
    nlp.add_pipe("scispacy_linker", config={"linker_name": "umls", "resolve_abbreviations": True})

    data_dir = "ACI/"
    file_list = []
    for root, ds, fs in os.walk(data_dir):
        for f in fs:
            file_list.append(os.path.join(root, f))
    print(file_list)

    medical_term_collection = []
    medical_term_map = {}
    for f in tqdm(file_list):
        with open(f, encoding='utf-8') as rf:
            for line in rf:
                source = json.loads(line)['source']
                summary = json.loads(line)['summary']

                source_ents, source_linked = entity_linker(nlp, source)
                medical_term_collection.extend(source_ents)
                for i in range(len(source_ents)):
                    if source_linked[i] not in medical_term_map.keys():
                        medical_term_map[source_linked[i]] = set([source_ents[i]])
                    else:
                        medical_term_map[source_linked[i]].add(source_ents[i])


                summary_ents, summary_linked = entity_linker(nlp, summary)
                medical_term_collection.extend(summary_ents)
                for i in range(len(summary_ents)):
                    if summary_linked[i] not in medical_term_map.keys():
                        medical_term_map[summary_linked[i]] = set([summary_ents[i]])
                    else:
                        medical_term_map[summary_linked[i]].add(summary_ents[i])

    medical_term_collection = list(set(medical_term_collection))
    num = 0
    for key in medical_term_map.keys():
        medical_term_map[key] = list(medical_term_map[key])
        num += len(medical_term_map[key])

    print(num)
    with open("ACI/ALL_medical_term_map.json", "w", encoding='utf-8') as wf:
        json.dump(medical_term_map, wf, ensure_ascii=False)
    with open("ACI/ALL_medical_term_file_train.txt", "w", encoding='utf-8') as wf:
        json.dump(medical_term_collection, wf, ensure_ascii=False)