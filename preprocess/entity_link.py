
import spacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector

import json
import os

THRESHOLD = 0


def entity_linker(document):
    nlp = spacy.load("en_core_sci_scibert")
    nlp.add_pipe("abbreviation_detector")
    nlp.add_pipe("scispacy_linker", config={"linker_name": "umls", "resolve_abbreviations": True})
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
    fp = "test.jsonl"
    with open(fp, encoding='utf-8') as rf:
        for lid, line in enumerate(rf):
            source = json.loads(line)['source']
            summary = json.loads(line)['summary']
            source_ents, source_linked = entity_linker(source)
            summary_ents, summary_linked = entity_linker(summary)
            with open("test_linked.jsonl", "a+", encoding='utf-8') as wf:
                res = {
                    "source": source,
                    "summary": summary,
                    "source_ents": source_ents,
                    "source_linked": source_linked,
                    "summary_ents": summary_ents,
                    "summary_linked": summary_linked
                }
                json.dump(res, wf, ensure_ascii=False)
                wf.write('\n')

