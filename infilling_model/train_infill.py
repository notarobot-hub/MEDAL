import argparse
import random
import json
import os
from tqdm import tqdm
from shutil import copyfile
from typing import Dict, Any
from pathlib import Path

import datasets
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, set_seed, get_linear_schedule_with_warmup

import logging

logger = logging.getLogger(__name__)

def add_model_specific_args(parser):
    parser.add_argument("--name", type=str, default='test', help="Name of expt")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--lr", type=float, default=0.00003, help="Maximum learning rate")
    parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--max_output_len", type=int, default=128, help="maximum num of wordpieces in the summary")
    parser.add_argument("--limit_val_batches", default=1.0, type=float, help='Percent of validation data used')
    parser.add_argument("--limit_test_batches", default=1.0, type=float, help='Percent of test data used')
    parser.add_argument("--limit_train_batches", default=1.0, type=float, help='Percent of training data used')
    parser.add_argument("--transformer_model", type=str, default='facebook/bart-base', help="transformer_model type")
    parser.add_argument("--output_dir", type=str, default='./saved_models/test', help="Location of output dir")
    parser.add_argument("--predict_file_path", type=str, default='./saved_models/test/test.json', help="Path for prediction data")
    parser.add_argument("--resume_checkpoint_dir", type=str, default="None", help="Location of resume ckpt")
    parser.add_argument("--resume_checkpoint_file", type=str, default="None", help="Filename of resume ckpt")
    parser.add_argument("--data_dir", type=str, default='data/', help="Location of input dir")
    parser.add_argument("--val_every", default=0.50, type=float, help='Validation every')
    parser.add_argument("--do_train", default=False, type=bool, help='Do training loop')
    parser.add_argument("--do_predict", default=False, type=bool, help='Do prediction loop')

    parser.add_argument("--max_input_len", type=int, default=1024, help="maximum num of wordpieces in the input")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps")
    parser.add_argument("--fp16", action='store_true', help="Use fp16 ")
    parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')

    return parser

SPECIAL_TOKEN_LIST = ["[MASK]"]

class SummarizationDataset(Dataset):
    def __init__(self, hf_arxiv_dataset, tokenizer, args):
        self.hf_arxiv_dataset = hf_arxiv_dataset
        self.tokenizer = tokenizer
        self.tokenizer.add_tokens(SPECIAL_TOKEN_LIST)
        self.args = args

    def __len__(self):
        return len(self.hf_arxiv_dataset)

    def __getitem__(self, idx):
        entry = self.hf_arxiv_dataset[idx]
        source = entry["masked_sent"] + " <SEP> " + entry["source"]
        target = entry["target"]

        input_ids = self.tokenizer.encode(source, truncation=True, max_length=self.args.max_input_len,
                                          padding='max_length')

        output_ids = self.tokenizer.encode(target, truncation=True, max_length=self.args.max_output_len,
                                           padding='max_length')

        return torch.tensor(input_ids), torch.tensor(output_ids)

    @staticmethod
    def collate_fn(batch):
        pad_token_id = 1
        input_ids, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, output_ids

    def process_example(self, entry):
        source = entry["masked_sent"] + " <SEP> " + entry['source_article_sentences']
        target = entry["target"]

        input_ids = self.tokenizer.encode(source, truncation=True, max_length=self.args.max_input_len,
                                          padding='max_length')

        output_ids = self.tokenizer.encode(target, truncation=True, max_length=self.args.max_output_len,
                                           padding='max_length')

        return torch.tensor(input_ids), torch.tensor(output_ids)


class Summarizer(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        self.args = params

        config = AutoConfig.from_pretrained(self.args.transformer_model)
        config.gradient_checkpointing = self.args.grad_ckpt
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.transformer_model, config=config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.transformer_model, use_fast=True)
        self.tokenizer.add_tokens(SPECIAL_TOKEN_LIST)
        self.model.resize_token_embeddings(len(self.tokenizer))

        if self.args.resume_checkpoint_dir != "None":
            saved_model = torch.load(os.path.join(self.args.resume_checkpoint_dir, self.args.resume_checkpoint_file))
            renamed_state_dict = {}
            for k, v in saved_model["state_dict"].items():
                new_key = k.replace("model.model.", "model.")
                renamed_state_dict[new_key] = v
            self.model = AutoModelForSeq2SeqLM.from_pretrained(None, config=config, state_dict=renamed_state_dict)

        self.rouge = datasets.load_metric('rouge')

    def forward(self, input_ids, output_ids):
        return self.model(input_ids,
                          attention_mask=(input_ids != self.tokenizer.pad_token_id),
                          labels=output_ids, use_cache=False)

    def training_step(self, batch, batch_nb):
        outputs = self.forward(*batch)
        epoch_num = self.current_epoch + 1
        self.log(f'train/train_step', batch_nb * epoch_num,
                 on_step=True, on_epoch=True)
        self.log('train/train_loss', outputs.loss, on_epoch=True)
        return {'loss': outputs.loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        dataset_size = len(self.hf_dataset['train'])
        gpu_count = torch.cuda.device_count()
        num_steps = dataset_size * self.args.epochs / gpu_count / self.args.grad_accum / self.args.batch_size
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup,
                                                    num_training_steps=num_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, split_name, is_train):
        dataset_split = self.hf_dataset[split_name]
        dataset = SummarizationDataset(hf_arxiv_dataset=dataset_split, tokenizer=self.tokenizer, args=self.args)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=SummarizationDataset.collate_fn)

    def train_dataloader(self):
        return self._get_dataloader('train', is_train=True)

    def val_dataloader(self):
        return self._get_dataloader('validation', is_train=False)

    def test_dataloader(self):
        return self._get_dataloader('test', is_train=False)

    def _evaluation_step(self, split, batch):
        input_ids, output_ids = batch
        generated_ids = self.model.generate(input_ids=input_ids,
                                            attention_mask=(input_ids != self.tokenizer.pad_token_id),
                                            use_cache=True, max_length=self.args.max_output_len, num_beams=1)

        predictions = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        references = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)

        # Compute rouge
        metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        results = self.rouge.compute(predictions=predictions, references=references)
        metrics = {}
        for metric_name in metric_names:
            metric_val = input_ids.new_zeros(1) + results[metric_name].mid.fmeasure
            metrics[f'{split}_{metric_name}'] = metric_val
        return metrics

    def _evaluation_epoch_end(self, split, step_outputs):
        metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        aggregated_metrics = {}
        for metric_name in metric_names:
            aggregated_metrics[f'{split}_{metric_name}'] = []

        for pred in step_outputs:
            for key, value in pred.items():
                aggregated_metrics[key].append(value)

        for key, value in aggregated_metrics.items():
            aggregated_metrics[key] = torch.mean(torch.stack(value, dim=0), dim=0, keepdim=False)
            self.log(f'{split}/{key}_epoch', aggregated_metrics[key], on_step=False, on_epoch=True, prog_bar=True,
                     sync_dist=True)
        return aggregated_metrics

    def validation_step(self, batch):
        return self._evaluation_step('val', batch)

    def validation_epoch_end(self, validation_step_outputs):
        fp = open(args.output_dir + "/val_metrics.txt", "a+")
        aggregated_metrics = self._evaluation_epoch_end('val', validation_step_outputs)
        for key, value in aggregated_metrics.items():
            fp.write(f'{key}_epoch: ' + str(aggregated_metrics[key]) + "\n")
        fp.write("\n")

    def _test_evaluation_step(self, split, batch):
        input_ids, output_ids = batch
        generated_ids = self.model.generate(input_ids=input_ids,
                                            attention_mask=(input_ids != self.tokenizer.pad_token_id),
                                            use_cache=True, max_length=self.args.max_output_len, num_beams=1)

        predictions = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        references = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)

        metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        results = self.rouge.compute(predictions=predictions, references=references)
        metrics = {}
        for metric_name in metric_names:
            metric_val = input_ids.new_zeros(1) + results[metric_name].mid.fmeasure
            metrics[f'{split}_{metric_name}'] = metric_val
        return metrics

    def test_step(self, batch):
        return self._test_evaluation_step('test', batch)

    def test_epoch_end(self, test_step_outputs):
        aggregated_metrics = self._evaluation_epoch_end('test', test_step_outputs)
        for key, value in aggregated_metrics.items():
            aggregated_metrics = value.cpu().item()
        fp = open(args.output_dir + "/metrics.json", "w")
        fp.write(json.dumps(aggregated_metrics))

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = Path(self.args.output_dir + "/hf_checkpoints/").joinpath(
            f"best_tfmr_step={self.trainer.global_step}")
        save_path.mkdir(exist_ok=True, parents=True)
        self.model.config.save_step = self.trainer.global_step
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    """
    Code reference from the work [Correcting Diverse Factual Errors in Abstractive Summarization via Post-Editing and Language Model Infilling] 
    """
    main_arg_parser = argparse.ArgumentParser(description="summarization")
    parser = add_model_specific_args(main_arg_parser)
    args = parser.parse_args()

    set_seed(args.seed)
    summarizer = Summarizer(args)

    if args.do_train:
        summarizer.hf_dataset = datasets.load_dataset('json',
                                                      data_files={"train": os.path.join(args.data_dir, "train.jsonl"),
                                                                  "validation": os.path.join(args.data_dir, "validation.jsonl"),
                                                                  "test": os.path.join(args.data_dir, "test.jsonl")})

    checkpoint_callback = ModelCheckpoint(monitor='val/val_rougeLsum_epoch',
                                          dirpath=args.output_dir,
                                          filename='tw-{epoch:02d}-{step}-val_rougeLsum_epoch{val/val_rougeLsum_epoch:.4f}',
                                          save_top_k=3,
                                          mode="max")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ckpt_path = None
    if args.resume_checkpoint_dir != "None":
        ckpt_path = os.path.join(args.resume_checkpoint_dir, args.resume_checkpoint_file)

    # Construct a PL trainer
    trainer = pl.Trainer(gpus=1,
                         accelerator='ddp',
                         max_epochs=args.epochs,
                         replace_sampler_ddp=False,
                         num_sanity_val_steps=0,
                         default_root_dir=args.output_dir,
                         precision=16 if args.fp16 else 32,
                         accumulate_grad_batches=args.grad_accum,
                         limit_val_batches=args.limit_val_batches,
                         limit_train_batches=args.limit_train_batches,
                         limit_test_batches=args.limit_test_batches,
                         callbacks=[checkpoint_callback, lr_monitor],
                         val_check_interval=args.val_every,
                         resume_from_checkpoint=ckpt_path,
                         track_grad_norm=2)
    if args.do_train:
        # Start training
        trainer.fit(summarizer)
        pa = checkpoint_callback.best_model_path
        best_ckpt_path = f'{args.output_dir}/best.ckpt'
        copyfile(pa, best_ckpt_path)

        # Start testing
        result = trainer.test(summarizer, ckpt_path=best_ckpt_path)
        print(result)

    if args.do_predict:
        print("Test on reference")

        dataset_split = datasets.load_dataset('json',
                                              data_files={"ref_test": args.predict_file_path})
        orig_test_dataset = SummarizationDataset(hf_arxiv_dataset=dataset_split, tokenizer=summarizer.tokenizer, args=summarizer.args)
        output_test_preds_file = os.path.join(args.output_dir+"/mask_cands/", args.predict_file_path.split("/")[-1])

        if torch.cuda.is_available():
            summarizer.model = summarizer.model.to(device=torch.device('cuda'))
        with open(output_test_preds_file, "w") as writer:
            for idx, entry in tqdm(enumerate(dataset_split["ref_test"])):
                if idx % 1000 == 0:
                    print("Processed "+str(idx)+" samples")
                input_ids, output_ids = orig_test_dataset.process_example(entry)
                input_ids = input_ids.unsqueeze(dim=0)

                input_ids = input_ids.to(summarizer.model.device)
                outputs = summarizer.model.generate(input_ids=input_ids,
                                                    attention_mask=(input_ids != summarizer.tokenizer.pad_token_id),
                                                    use_cache=False, max_length=args.max_output_len, num_beams=15, num_return_sequences=15,
                                                    return_dict_in_generate=True, output_hidden_states=True, early_stopping=True)
                generated_ids = outputs["sequences"]

                predictions = summarizer.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
                target_words = entry["target"].split()
                cand_preds = [x for x in predictions[5:] if not (x in entry["target"] or entry["target"] in x or len(list(set(entry["target"].split()) & set(x.split())))>=min(len(target_words)/2, len(x.split())/2) )]
                for pid, pred in enumerate(cand_preds):
                    replaced_summary_sent = entry["masked_sent"].replace("[blank]", pred)
                    label = 0
                    err_span = pred
                    corr_span = entry["target"]
                    err_type = "Mask Model Cand"
                    error_prob = random.random()
                    if error_prob < 0.4:
                        replaced_summary_sent = entry["original_sent"]
                        label  = 1
                        err_type = "None"
                        err_span = "None"
                        corr_span = "None"

                    error_summary_sents = []
                    chosen_sent_idx = None
                    for sid, x in enumerate(entry["original_summary_sentences"]):
                        if x == entry["original_sent"]:
                            error_summary_sents.append(replaced_summary_sent)
                            chosen_sent_idx = sid
                        else:
                            error_summary_sents.append(x)
                    data = {"source_article_sentences": entry["source_article_sentences"],  # str
                            "original_summary_sentences": entry["original_summary_sentences"],  # List
                            "generated_summary_sentences": error_summary_sents,  # List
                            "incorrect_sent_idx": chosen_sent_idx,
                            "original_summary_sent": entry["original_sent"],
                            "generated_summary_sent": replaced_summary_sent,
                            "label": label,
                            "error_type": err_type,
                            "err_span": err_span,
                            "corr_span": corr_span}
                    writer.write(json.dumps(data) + "\n")
