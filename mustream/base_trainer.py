"""Base Trainer class for training model via PyTorch Lightning."""

# References:
# https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/fabric/build_your_own_trainer/trainer.py
# https://lightning.ai/docs/pytorch/stable/common/trainer.html
# https://lightning.ai/docs/pytorch/stable/model/train_model_basic.html
# https://lightning.ai/docs/fabric/stable/guide/lightning_module.html

import lightning as L
import torch
from typing import Optional

from pathlib import Path
from lightning.fabric.strategies import FSDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.utilities import grad_norm


class ValEndModelCheckpoint(ModelCheckpoint):
    """`ModelCheckpoint` that ranks a step's checkpoint on that step's own val metrics.

    Lightning's `every_n_train_steps` trigger lives in `on_train_batch_end`, which
    the training epoch loop calls *before* running the validation for that same step
    (`training_epoch_loop.advance` -> `on_train_batch_end`, then `on_advance_end` ->
    `val_loop.run()`). With val_interval=1000 and checkpoint_interval=5000 that has
    two consequences, both of which cost us a checkpoint:

      * the file written at step N is ranked on the metric measured at step
        N - val_interval, so its name and its score come from different weights.
        In one 330M run, step=90000.ckpt recorded val/total_loss 0.3647, which is
        step *89000*'s value; step 90000 itself measured 0.3653.
      * the last validation of a run, at max_steps, is never ranked at all --
        training ends before the next `on_train_batch_end` -- so a final-step
        improvement can never win. In that same run this discarded the minimum of
        the entire curve (0.36428 at step 100000, against the 0.3647 kept).

    Only the *timing* of the comparison changes here: the same "every
    every_n_train_steps steps" test is moved to `on_validation_end`, where
    `trainer.callback_metrics` holds the current step's metrics. The number of
    candidates, the top-k bookkeeping and the {step} filename are untouched, so
    checkpoint_interval keeps its meaning -- it just has to stay a multiple of
    val_interval (enforced in Trainer.__init__) or the save grid and the validation
    grid never meet and nothing is written.
    """

    def __init__(self, *args, every_n_train_steps: int = 0, **kwargs):
        # Both of Lightning's own triggers off: `on_train_batch_end` returns early
        # on every_n_train_steps=0, and the top-k branch of the base
        # `on_validation_end` requires every_n_epochs >= 1. Saving is driven purely
        # by the override below.
        super().__init__(*args, every_n_train_steps=0, every_n_epochs=0, **kwargs)
        self._save_every_n_steps = every_n_train_steps

    @property
    def state_key(self) -> str:
        # Deliberately the key a plain ModelCheckpoint(every_n_train_steps=N) would
        # produce. Lightning matches saved callback state by this key and silently
        # skips a mismatch, so keeping it lets `load_from_latest_checkpoint` resume
        # runs written before this class existed with their best_model_score and
        # best_k_models intact instead of starting the top-k bookkeeping over.
        return "{}{!r}".format(
            ModelCheckpoint.__qualname__,
            {
                "monitor": self.monitor,
                "mode": self.mode,
                "every_n_train_steps": self._save_every_n_steps,
                "every_n_epochs": self._every_n_epochs,
                "train_time_interval": self._train_time_interval,
            },
        )

    def on_validation_end(self, trainer, pl_module) -> None:
        if self._save_every_n_steps < 1 or trainer.global_step % self._save_every_n_steps:
            return
        # Covers sanity checking (global_step 0 passes the test above), non-fit
        # stages, and a step already saved.
        if self._should_skip_saving_checkpoint(trainer):
            return
        monitor_candidates = self._monitor_candidates(trainer)
        # A missing checkpoint_metric raises out of here rather than being ignored,
        # since the validation loop has run by now and the metric should be present.
        self._save_topk_checkpoint(trainer, monitor_candidates)
        self._save_last_checkpoint(trainer, monitor_candidates)


class Trainer:
    """Step-based trainer around a Lightning `L.Trainer`.

    Takes the Lightning module and its dataloaders, sets up wandb logging and
    step-based checkpointing in `save_dir`, and resumes from the newest
    checkpoint there when `load_from_latest_checkpoint` is set.
    """

    def __init__(
        self,
        args: dict,
        lit_module: L.LightningModule,
        save_dir: str,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        seed: int = 42,
        # Training args
        train_steps: int = 1000,
        val_interval: int = 1000,
        # You need to define sample_interval in module if you want to use it
        limit_val_batches: Optional[int] = None,
        overfit_batches: int = 0,  # set to 1 for debugging
        # Logging args
        wandb_project: str = "mustream",
        log_every_n_steps: int = 1,
        # Checkpointing args
        checkpoint_interval: int = 0,  # set to 0 to disable
        checkpoint_metric: str = "val/loss",
        checkpoint_mode: str = "min",
        checkpoint_top_k: int = 2,
        load_from_latest_checkpoint: bool = False,
        checkpoint_path: str = None,
        # Fabric/distribution training args
        use_fabric: bool = False,
        accelerator: str = "auto",
        devices: str = "auto",
        num_nodes: int = 1,
        precision: str = None,
        strategy: str = "auto",
        # Gradient clipping args
        gradient_clip_val: float = 1.0,
    ):
        super().__init__()
        self.args = args  # the overall arguments
        self.save_dir = Path(save_dir)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # For GPUs with Tensor Cores
        torch.set_float32_matmul_precision("high")

        if use_fabric:
            # Setup distribution training
            if strategy == "fsdp":
                strategy = FSDPStrategy(state_dict_type="full")
            self.fabric = L.Fabric(
                accelerator=accelerator,
                num_nodes=num_nodes,
                precision=precision,
                devices=devices,
                strategy=strategy,
            )
            self.fabric.launch()
            self.lit_module = self.fabric.setup(lit_module)
            self.fabric.barrier()
            self.fabric.seed_everything(seed)
            self.distributed = True
            # No wandb logging for processes other than rank 0
            self.wandb_offline = not self.fabric.is_global_zero

        else:
            # seed_everything(seed, workers=True)  # `workers=True` for reproducibility
            seed_everything(seed)
            self.lit_module = lit_module
            self.distributed = False
            self.wandb_offline = False

        self.ckpt_path = None
        if self.save_dir.exists() and load_from_latest_checkpoint:
            ckpt_files = list(self.save_dir.glob("*.ckpt"))
            if ckpt_files:
                latest_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
                print(f"Loading checkpoint from {latest_ckpt}")
                self.ckpt_path = latest_ckpt
        if checkpoint_path is not None:
            self.ckpt_path = checkpoint_path

        # Checkpointing. ValEndModelCheckpoint ranks a checkpoint at the end of the
        # validation run belonging to its own step, so the two intervals have to land
        # on the same steps.
        if checkpoint_interval and isinstance(val_interval, int) and checkpoint_interval % val_interval:
            raise ValueError(
                f"checkpoint_interval ({checkpoint_interval}) must be a multiple of "
                f"val_interval ({val_interval}); otherwise no step is ever both a "
                "validation step and a checkpoint step, and nothing is saved."
            )
        checkpoint_callback = ValEndModelCheckpoint(
            dirpath=self.save_dir,
            filename="{step}",  # save by step
            every_n_train_steps=checkpoint_interval,
            monitor=checkpoint_metric,
            mode=checkpoint_mode,
            save_top_k=checkpoint_top_k,
            enable_version_counter=False,  # overwrite the same file
        )

        # Log learning rate
        lr_monitor = LearningRateMonitor(logging_interval="step")

        # Setup the logger
        wandb_id = None
        if self.ckpt_path:
            wandb_dir = self.save_dir / "wandb" / "latest-run"
            if wandb_dir.exists():
                # get the wandb id from the checkpoint path and resume the run
                wandb_file = list(wandb_dir.glob("*.wandb"))[0]
                wandb_id = wandb_file.stem.split("-")[-1]

        logger = WandbLogger(
            name=self.save_dir.name,
            project=wandb_project,
            log_model=False,
            save_dir=self.save_dir,
            offline=self.wandb_offline,
            config=args,
            id=wandb_id,
        )

        self.trainer = L.Trainer(
            max_steps=train_steps,
            val_check_interval=val_interval,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            callbacks=[
                checkpoint_callback,
                lr_monitor,
            ],
            limit_val_batches=limit_val_batches,
            overfit_batches=overfit_batches,
            check_val_every_n_epoch=None,
            accelerator=accelerator,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            strategy=strategy,
            gradient_clip_val=gradient_clip_val,
            # deterministic=True  # for reproducibility
        )

    def train(self):
        """The entry point for training the model."""
        self.trainer.fit(
            self.lit_module,
            self.train_dataloader,
            self.val_dataloader,
            ckpt_path=self.ckpt_path,
        )
        
    def test(self):
        """Run the test loop on the validation dataloader."""
        self.trainer.test(
            self.lit_module,
            self.val_dataloader,
            ckpt_path=self.ckpt_path,
        )


class BaseLightningModel(L.LightningModule):
    """Base Lightning module for the MERT distillation modules.

    Subclasses define `self.model` (used for grad-norm logging) and set
    `self.train_dataloader` / `self.val_dataloader`.
    """

    def training_step(self, batch, batch_idx):
        """
        A single step during training. Calculates the loss for the batch and logs it.
        """
        pass

    def validation_step(self, batch, batch_idx):
        """
        A single step during validation. Calculates the loss for the batch and logs it.
        """
        pass

    def test_step(self, batch, batch_idx):
        """
        A single step during testing. Calculates the loss for the batch and logs it.
        """
        pass

    def configure_optimizers(self):
        """
        Configures and returns the optimizer(s).
        """
        pass

    def on_train_start(self):
        """
        Hook that is called at the very beginning of training (before any epochs).
        You can set up things like timers, experiment tracking, etc. here.
        """
        pass

    def on_train_end(self):
        """
        Hook that is called at the end of training (after all epochs).
        This is useful for cleanup, final logging, saving models, etc.
        """
        pass

    def get_dataloaders(self):
        return self.train_dataloader, self.val_dataloader

    def _log_scalar(self, name, value, **kwargs):
        self.log(
            name,
            value,
            prog_bar=True,
            logger=True,
            **kwargs,
        )

    def _log_dict(self, dict_, **kwargs):
        self.log_dict(dict_, prog_bar=True, logger=True, **kwargs)

    def on_before_optimizer_step(self, optimizer):
        """
        Compute the 2-norm for each layer.
        If using mixed precision, the gradients are already unscaled here
        """
        norms = grad_norm(self.model, norm_type=2)
        self._log_dict({"grad_norm": norms["grad_2.0_norm_total"]})
        # print(f"global_step: {self.global_step}")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch["waveform"] = batch["waveform"].to(device)
        return batch