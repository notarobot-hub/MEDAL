from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from fsspec.core import url_to_fs


class UniversalEarlyStopping(EarlyStopping):
    def __init__(self, args):
        super().__init__(
            monitor=args.monitor,
            mode=args.mode,
            patience=args.patience
        )


class UniversalCheckpoint(ModelCheckpoint):
    def __init__(self, args):
        super().__init__(
            monitor=args.monitor,
            save_top_k=args.save_top_k,
            mode=args.mode,
            filename=f"{args.model_name}_lr{args.learning_rate}_ep{args.max_epochs}_w{args.warm_steps}" + args.ckpt_filename,
            save_last=True,
        )

