import copy
import re

import nltk
import torch
import os
from torch.utils.data import Dataset
from rouge import Rouge

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CorrDataset(Dataset):
    def __init__(self, args, dataset, tokenizer):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.encode(self.dataset[item], item)

    def encode(self, data, idx):
        if self.args.do_train:
            orig_summ_sent = data['original_summary_sent']
            gen_summ_sent = data['generated_summary_sent']
            source_article = data['source_article_sentences']

            input_text = gen_summ_sent + " <SEP> " + source_article
            input = self.tokenizer(input_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            target_text = orig_summ_sent

            if self.args.generate_factuality_label:
                target_text = str(data['label']) + " <SEP> " + target_text

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(target_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            return {
                "source_ids": torch.tensor(input["input_ids"]),
                "source_mask": torch.tensor(input["attention_mask"]),
                "target_ids": torch.tensor(labels["input_ids"]),
                "source_article": source_article,
                "indice": torch.tensor(idx)
            }

        if self.args.do_test:
            orig_summ_sent = data['original_summary_sentences']
            gen_summ_sent = data['generated_summary_sentences']
            source_article = data['source_article_sentences']
            indice = data['source_id']

            input_text = gen_summ_sent + " <SEP> " + source_article
            input = self.tokenizer(input_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            return {
                "source_ids": torch.tensor(input["input_ids"]),
                "source_mask": torch.tensor(input["attention_mask"]),
                "source_article": source_article,
                "target_summary":orig_summ_sent,
                "indice": torch.tensor(indice)
            }


class CorrInstructionDataset(Dataset):
    def __init__(self, args, dataset, tokenizer):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.encode(self.dataset[item], item)

    def encode(self, data, idx):
        instruction_text = "Check for any hallucinations in the summary, and if found, correct them."
        if self.args.do_train:
            orig_summ_sent = data['original_summary_sent']
            gen_summ_sent = data['generated_summary_sent'] + "\n\n"
            source_article = data['source_article_sentences'] + "\n\n"

            input_text = instruction_text + "Text: " + source_article + "Summary: " + gen_summ_sent
            input = self.tokenizer(input_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            if self.args.generate_factuality_label:
                if data['label'] == 0:
                    target_text = "[Yes]\n\n" + orig_summ_sent
                else:
                    target_text = "[No]\n\n"
            else:
                target_text = orig_summ_sent

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(target_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            return {
                "source_ids": torch.tensor(input["input_ids"]),
                "source_mask": torch.tensor(input["attention_mask"]),
                "target_ids": torch.tensor(labels["input_ids"]),
                "source_article": source_article,
                "indice": torch.tensor(idx)
            }

        if self.args.do_test:
            orig_summ_sent = data['original_summary_sentences']
            gen_summ_sent = data['generated_summary_sentences'] + "\n\n"
            source_article = data['source_article_sentences'] + "\n\n"
            indice = data['source_id']

            input_text = instruction_text + "Text: " + source_article + "Summary: " + gen_summ_sent
            input = self.tokenizer(input_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            return {
                "source_ids": torch.tensor(input["input_ids"]),
                "source_mask": torch.tensor(input["attention_mask"]),
                "source_article": source_article,
                "target_summary":orig_summ_sent,
                "indice": torch.tensor(indice)
            }

class ACIDataset(Dataset):
    def __init__(self, args, dataset, tokenizer):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.encode(self.dataset[item], item)

    def encode(self, data, idx):
        instruction_text = "Check for any hallucinations in the summary, and if found, correct them."

        def sent_tokenizer(sent):
            sentences = re.findall(r'\[(?:doctor|patient)\][^[]*', sent)
            sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
            return sentences

        # select related sentences
        def gen_relative(source, sum):
            rouge = Rouge()
            sum_list = sent_tokenizer(sum)
            source_list = sent_tokenizer(source)
            res = set()
            for s1 in sum_list:
                max_rouge = -1
                cur_src = ""
                for s2 in source_list:
                    # print("sum: " + s1)
                    # print("source: " + s2)
                    try:
                        rouge_score = rouge.get_scores(s1, s2)[0]["rouge-l"]["f"]
                    except:
                        print("sum: " + s1)
                        print("source: " + s2)

                    # print(rouge_score)
                    if rouge_score > max_rouge:
                        max_rouge = rouge_score
                        cur_src = s2
                res.add(cur_src)
            return " ".join(list(res))



        if self.args.do_train:
            orig_summ_sent = data['original_summary_sent']
            gen_summ_sent = data['generated_summary_sent'] + "\n\n"
            source_article = data['source_article_sentences'] + "\n\n"

            relative_article = gen_relative(source_article, orig_summ_sent)

            input_text = instruction_text + "Text: " + relative_article + "Summary: " + gen_summ_sent
            input = self.tokenizer(input_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            if self.args.generate_factuality_label:
                if data['label'] == 0:
                    target_text = "[Yes]\n\n" + orig_summ_sent
                else:
                    target_text = "[No]\n\n"
            else:
                target_text = orig_summ_sent

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(target_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            return {
                "source_ids": torch.tensor(input["input_ids"]),
                "source_mask": torch.tensor(input["attention_mask"]),
                "target_ids": torch.tensor(labels["input_ids"]),
                "source_article": source_article,
                "indice": torch.tensor(idx)
            }

        if self.args.do_test:
            orig_summ_sent = data['original_summary_sentences']
            gen_summ_sent = data['generated_summary_sentences'] + "\n\n"
            source_article = data['source_article_sentences'] + "\n\n"
            indice = data['source_id']

            relative_article = gen_relative(source_article, orig_summ_sent)

            input_text = instruction_text + "Text: " + relative_article + "Summary: " + gen_summ_sent
            input = self.tokenizer(input_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            return {
                "source_ids": torch.tensor(input["input_ids"]),
                "source_mask": torch.tensor(input["attention_mask"]),
                "source_article": source_article,
                "target_summary":orig_summ_sent,
                "indice": torch.tensor(indice)
            }


class HQSDataset(Dataset):
    def __init__(self, args, dataset, tokenizer):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.encode(self.dataset[item])

    def encode(self, data):
        source_text = data['source']
        target_text = data['summary']

        source = self.tokenizer.encode_plus(source_text, max_length=self.args.max_input_length, padding="max_length",
                                            truncation=True, return_tensors="pt")
        target = self.tokenizer.encode_plus(target_text, max_length=self.args.max_output_length, padding="max_length",
                                            truncation=True, return_tensors="pt")

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()

        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }


class TyDiQADataset(Dataset):
    def __init__(self, args, dataset, tokenizer):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.encode(self.dataset[item], item)

    def encode(self, data, idx):
        instruction_text = "Answer the following question based on the given context. If the question cannot be answered from the context, say 'I cannot answer this question based on the given context.'"
        
        if self.args.do_train:
            question = data['question']
            context = data['context']
            answer = data['answer']
            
            input_text = instruction_text + "\n\nContext: " + context + "\n\nQuestion: " + question
            input = self.tokenizer(input_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            if self.args.generate_factuality_label:
                # For TyDiQA, we can use answer presence as a factuality indicator
                if answer and answer.strip():
                    target_text = "[Yes]\n\n" + answer
                else:
                    target_text = "[No]\n\nI cannot answer this question based on the given context."
            else:
                target_text = answer if answer else "I cannot answer this question based on the given context."

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(target_text, max_length=self.args.max_output_length, padding="max_length", truncation=True)

            return {
                "source_ids": torch.tensor(input["input_ids"]),
                "source_mask": torch.tensor(input["attention_mask"]),
                "target_ids": torch.tensor(labels["input_ids"]),
                "context": context,
                "indice": torch.tensor(idx)
            }

        if self.args.do_test:
            question = data['question']
            context = data['context']
            answer = data.get('answer', '')
            indice = data.get('id', idx)

            input_text = instruction_text + "\n\nContext: " + context + "\n\nQuestion: " + question
            input = self.tokenizer(input_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            return {
                "source_ids": torch.tensor(input["input_ids"]),
                "source_mask": torch.tensor(input["attention_mask"]),
                "context": context,
                "target_answer": answer,
                "indice": torch.tensor(indice)
            }


class TriviaQADataset(Dataset):
    def __init__(self, args, dataset, tokenizer):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.encode(self.dataset[item], item)

    def encode(self, data, idx):
        instruction_text = "Answer the following trivia question based on the given context. Provide a concise and accurate answer."
        
        if self.args.do_train:
            question = data['question']
            context = data['context']
            answer = data['answer']
            
            input_text = instruction_text + "\n\nContext: " + context + "\n\nQuestion: " + question
            input = self.tokenizer(input_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            if self.args.generate_factuality_label:
                # For TriviaQA, we can use answer presence as a factuality indicator
                if answer and answer.strip():
                    target_text = "[Yes]\n\n" + answer
                else:
                    target_text = "[No]\n\nI cannot answer this question based on the given context."
            else:
                target_text = answer if answer else "I cannot answer this question based on the given context."

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(target_text, max_length=self.args.max_output_length, padding="max_length", truncation=True)

            return {
                "source_ids": torch.tensor(input["input_ids"]),
                "source_mask": torch.tensor(input["attention_mask"]),
                "target_ids": torch.tensor(labels["input_ids"]),
                "context": context,
                "indice": torch.tensor(idx)
            }

        if self.args.do_test:
            question = data['question']
            context = data['context']
            answer = data.get('answer', '')
            indice = data.get('id', idx)

            input_text = instruction_text + "\n\nContext: " + context + "\n\nQuestion: " + question
            input = self.tokenizer(input_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            return {
                "source_ids": torch.tensor(input["input_ids"]),
                "source_mask": torch.tensor(input["attention_mask"]),
                "context": context,
                "target_answer": answer,
                "indice": torch.tensor(indice)
            }


class TruthfulQADataset(Dataset):
    def __init__(self, args, dataset, tokenizer):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.encode(self.dataset[item], item)

    def encode(self, data, idx):
        instruction_text = "Answer the following question truthfully. If you're not sure about the answer, say 'I'm not sure about this.'"
        
        if self.args.do_train:
            question = data['question']
            answer = data['answer']
            
            input_text = instruction_text + "\n\nQuestion: " + question
            input = self.tokenizer(input_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            if self.args.generate_factuality_label:
                # For TruthfulQA, we can use the provided truthfulness label
                if data.get('label', 1) == 1:  # Assuming 1 means truthful
                    target_text = "[Yes]\n\n" + answer
                else:
                    target_text = "[No]\n\n" + answer
            else:
                target_text = answer

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(target_text, max_length=self.args.max_output_length, padding="max_length", truncation=True)

            return {
                "source_ids": torch.tensor(input["input_ids"]),
                "source_mask": torch.tensor(input["attention_mask"]),
                "target_ids": torch.tensor(labels["input_ids"]),
                "question": question,
                "indice": torch.tensor(idx)
            }

        if self.args.do_test:
            question = data['question']
            answer = data.get('answer', '')
            indice = data.get('id', idx)

            input_text = instruction_text + "\n\nQuestion: " + question
            input = self.tokenizer(input_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            return {
                "source_ids": torch.tensor(input["input_ids"]),
                "source_mask": torch.tensor(input["attention_mask"]),
                "question": question,
                "target_answer": answer,
                "indice": torch.tensor(indice)
            }


class CoQADataset(Dataset):
    def __init__(self, args, dataset, tokenizer):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.encode(self.dataset[item], item)

    def encode(self, data, idx):
        instruction_text = "Answer the following question based on the given context. If the question cannot be answered from the context, say 'I cannot answer this question based on the given context.'"
        
        if self.args.do_train:
            question = data['question']
            context = data['context']
            answer = data['answer']
            
            input_text = instruction_text + "\n\nContext: " + context + "\n\nQuestion: " + question
            input = self.tokenizer(input_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            if self.args.generate_factuality_label:
                # For CoQA, we can use answer presence as a factuality indicator
                if answer and answer.strip():
                    target_text = "[Yes]\n\n" + answer
                else:
                    target_text = "[No]\n\nI cannot answer this question based on the given context."
            else:
                target_text = answer if answer else "I cannot answer this question based on the given context."

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(target_text, max_length=self.args.max_output_length, padding="max_length", truncation=True)

            return {
                "source_ids": torch.tensor(input["input_ids"]),
                "source_mask": torch.tensor(input["attention_mask"]),
                "target_ids": torch.tensor(labels["input_ids"]),
                "context": context,
                "indice": torch.tensor(idx)
            }

        if self.args.do_test:
            question = data['question']
            context = data['context']
            answer = data.get('answer', '')
            indice = data.get('id', idx)

            input_text = instruction_text + "\n\nContext: " + context + "\n\nQuestion: " + question
            input = self.tokenizer(input_text, max_length=self.args.max_input_length, padding="max_length", truncation=True)

            return {
                "source_ids": torch.tensor(input["input_ids"]),
                "source_mask": torch.tensor(input["attention_mask"]),
                "context": context,
                "target_answer": answer,
                "indice": torch.tensor(indice)
            }

