"""Entry point for MERT *causal* distillation -- turning a frozen
bidirectional MERT into a streaming (block-causal, KV-cache) student
(modeling/streaming/streaming_mert.py), under a cosine + L2 feature loss.

    python train_mert_causal_distill.py --args.load=config/mert_causal_distill.yml
"""
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
torch._dynamo.config.disable = True

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import argbind

from functools import partial
from pathlib import Path

from base_trainer import Trainer
from lit_module.mert_causal_distill import LitMERTCausalDistill

GROUP = __file__
# Binding things only when this file is loaded
bind = partial(argbind.bind, group=GROUP)
Trainer = bind(Trainer, without_prefix=True)


@bind(without_prefix=True)
def main(args, save_dir: str = ""):

    # Create the lit module
    lit_module = LitMERTCausalDistill()

    # Get the dataloaders
    train_dataloader, val_dataloader = lit_module.get_dataloaders()

    # Train the model
    trainer = Trainer(
        args=args,
        lit_module=lit_module,
        save_dir=save_dir,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        load_from_latest_checkpoint=True
    )
    trainer.train()


if __name__ == "__main__":
    args = argbind.parse_args(group=GROUP)
    argbind.dump_args(args, Path(args["save_dir"]) / "args.yml")
    with argbind.scope(args):
        main(args)
