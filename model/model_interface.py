import math
import os

import datasets
import nltk
import numpy as np
import pytorch_lightning as pl
import torch
from transformers import get_linear_schedule_with_warmup

from utils.data_utils import write_pred_data, write_corr_pred_data


class MInterface(pl.LightningModule):
    def __init__(self, args, tokenizer, task_model):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.model = task_model
        self.text_path = os.path.join(args.data_dir)

    def setup(self, stage=None) -> None:
        if stage == 'fit':
            num_gpus = self.trainer.gpus if self.trainer.gpus is not None else 0
            self.total_step = int(
                self.trainer.max_epochs * self.num_data / (max(1, num_gpus) * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(params, lr=self.args.learning_rate)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=self.args.warm_steps, num_training_steps=self.total_step)

        return [{
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'interval': 'step',
            #     'frequency': 1
            # }
        }]

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        outputs = self.model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
        )
        if torch.isnan(outputs.loss).any():
            print(batch['source_ids'].shape)
            print(batch['target_ids'].shape)

        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"loss": loss}

    def training_epoch_end(self, train_step_outputs):
        avg_train_loss = torch.stack([x['loss'] for x in train_step_outputs]).mean()
        print(avg_train_loss)
        self.log("train_loss", avg_train_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"loss": loss}

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.stack([x["loss"] for x in val_step_outputs]).mean()
        print(avg_val_loss)
        self.log("val_loss", avg_val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        labels = batch["target_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        print(batch["source_ids"].size())
        print(batch["source_mask"].size())
        print(batch["target_mask"].size())
        predictions_ids = self.model.generate(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            repetition_penalty=1.5,
            num_beams=4,
            max_length=self.args.max_output_length,
            length_penalty=0.8,
            early_stopping=True
        )

        # repetition_penalty > 1
        predictions = predictions_ids.cpu().numpy()
        predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return {
            "predictions": predictions
        }

    def test_epoch_end(self, test_step_outputs):
        predictions = [item for pred in test_step_outputs for item in pred['predictions']]
        write_pred_data(self.args, self.text_path, predictions)

class CorrInterface(pl.LightningModule):
    def __init__(self, args, tokenizer, task_model):
        super().__init__()
        self.args = args

        self.tokenizer = tokenizer
        # special_tokens_dict = {'additional_special_tokens': ["<SEP>"]}
        # self.tokenizer.add_special_tokens(special_tokens_dict)

        self.model = task_model
        # self.model.resize_token_embeddings(len(tokenizer))

        self.text_path = os.path.join(args.data_dir)

        self.rouge = datasets.load_metric('rouge')

    def setup(self, stage=None) -> None:
        if stage == 'fit':
            num_gpus = self.trainer.gpus if self.trainer.gpus is not None else 0
            self.total_step = int(
                self.trainer.max_epochs * self.num_data / (max(1, num_gpus) * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(params, lr=self.args.learning_rate)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=self.args.warm_steps, num_training_steps=self.total_step)

        return [{
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'interval': 'step',
            #     'frequency': 1
            # }
        }]

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        outputs = self.model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
        )
        if torch.isnan(outputs.loss).any():
            print(batch['source_ids'].shape)
            print(batch['target_ids'].shape)

        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"loss": loss}

    def training_epoch_end(self, train_step_outputs):
        avg_train_loss = torch.stack([x['loss'] for x in train_step_outputs]).mean()
        print(avg_train_loss)
        self.log("train_loss", avg_train_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"loss": loss}

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.stack([x["loss"] for x in val_step_outputs]).mean()
        print(avg_val_loss)
        self.log("val_loss", avg_val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        outputs = self.model.generate(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
            repetition_penalty=1.5,
            num_beams=4,
            max_length=self.args.max_output_length,
            length_penalty=0.8,
            early_stopping=True)
        generated_ids = outputs["sequences"]
        predictions = self.tokenizer.batch_decode(generated_ids.tolist())
        goldens = self.tokenizer.batch_decode(batch["source_ids"])

        preds = ["\n".join(nltk.sent_tokenize(pred.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()))
                 for pred in predictions]
        golds = ["\n".join(nltk.sent_tokenize(gold.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()))
                 for gold in goldens]
        gold_sums = []
        for gold in golds:
            gold_sums.append(gold.split('\n')[-1].replace("Summary: ", ""))

        ref_preds = []
        if self.args.task == "Corr":
            if self.args.generate_factuality_label:
                for pred in preds:
                    pred = pred.split(" <SEP> ")[-1]
                    ref_preds.append(pred)
            else:
                ref_preds = preds
        elif self.args.task == "CorrInstruction":
            for pid, pred in enumerate(preds):
                label = pred.split(' ')[0]
                if label == "[Yes]":
                    ref_preds.append(" ".join(pred.split(' ')[1:]))
                else:
                    ref_preds.append(gold_sums[pid])
        return ref_preds

    def test_epoch_end(self, test_step_outputs):
        predictions = []
        for pred in test_step_outputs:
            # print(pred)
            predictions.extend(pred)
        write_corr_pred_data(self.args, predictions)
