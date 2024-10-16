import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration

from config import Config
from model import MInterface, CorrInterface
from data import DInterface

from utils.callbacks import UniversalCheckpoint, UniversalEarlyStopping
from data.dataset import HQSDataset, CorrDataset, CorrInstructionDataset, ACIDataset

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main(args):
    pl.seed_everything(args.seed)

    if args.model_name == "pegasus-large":
        tokenizer = PegasusTokenizer.from_pretrained(args.model_path)
        model = PegasusForConditionalGeneration.from_pretrained(args.model_path)
    elif args.model_name == "biobart-large":
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    elif args.model_name == "flan-t5-base":
        tokenizer = T5Tokenizer.from_pretrained(args.model_path)
        model = T5ForConditionalGeneration.from_pretrained(args.model_path)

    if args.task == "Corr":
        data_module = DInterface(args, tokenizer, task_dataset=CorrDataset)
    elif args.task == "CorrInstruction":
        if args.dataset == "ACI":
            data_module = DInterface(args, tokenizer, task_dataset=ACIDataset)
        else:
            data_module = DInterface(args, tokenizer, task_dataset=CorrInstructionDataset)

    # baseline_stanford experiments
    if args.task == "HQS" or args.task == "RRS":
        data_module = DInterface(args, tokenizer, task_dataset=HQSDataset)


    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if "Corr" not in args.task:
        logger = TensorBoardLogger(save_dir=args.log_dir, name=args.model_type, version=args.load_v_num)
    else:
        logger = TensorBoardLogger(save_dir=args.log_dir, name=args.corr_model_type, version=args.load_v_num)

    callbacks = [UniversalCheckpoint(args), UniversalEarlyStopping(args)]

    trainer = Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)

    if args.task == "Corr" or args.task == "CorrInstruction":
        model_interface = CorrInterface(args, tokenizer, model)
    else:
        model_interface = MInterface(args, tokenizer, model)
    model_interface.num_data = len(data_module.train_dataset)

    if args.do_train:
        trainer.fit(model_interface, data_module.train_dataloader(), data_module.val_dataloader())

    if args.do_test:
        args.ckpt_path = "best.ckpt"
        trainer.test(model_interface, data_module.test_dataloader(), ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/HQS_Pegasus.json')
    parser.add_argument('--dataset', default='HQS', type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    args = parser.parse_args()
    args = Config(args)

    print(args)

    main(args)
