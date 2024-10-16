import os

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from utils.data_utils import load_data
from .dataset import HQSDataset

class DInterface(pl.LightningDataModule):
    def __init__(self, args, tokenizer, task_dataset):
        super().__init__()
        self.args = args

        self.tokenizer = tokenizer
        # special_tokens_dict = {'additional_special_tokens': ["<SEP>"]}
        # self.tokenizer.add_special_tokens(special_tokens_dict)

        self.train_dataset, self.dev_dataset, self.test_dataset = load_data(args, args.data_dir)

        self.train_set = task_dataset(self.args, self.train_dataset, self.tokenizer)
        self.dev_set = task_dataset(self.args, self.dev_dataset, self.tokenizer)
        self.test_set = task_dataset(self.args, self.test_dataset, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.args.train_batch_size, num_workers=self.args.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dev_set, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, shuffle=False)

