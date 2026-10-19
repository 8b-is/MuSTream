"""Linear warmup + cosine decay learning rate scheduler."""

from typing import List

import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class LinearWarmupCosineDecay(_LRScheduler):
    """
    Learning rate scheduler that combines a linear warmup + cosine decay.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_iters (int): Number of iterations for the linear warmup phase.
        total_iters (int): Total number of iterations for the entire schedule.
        eta_min (float, optional): Minimum learning rate after cosine decay. Default is 0.
        last_iter (int, optional): The index of the last iteration. Default is -1.
        restart (bool, optional): Ignore the scheduler state stored in a
            resumed checkpoint and start a fresh warmup + cosine cycle from
            iteration 0 (see load_state_dict). Default is False.

    Methods:
        get_lr() -> List[float]:
            Compute the learning rate for the current iteration.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iters: int = 0,
        total_iters: int = 0,
        eta_min: float = 0,
        last_iter: int = -1,
        restart: bool = False,
    ):
        if warmup_iters <= 0:
            raise ValueError("Warmup iterations must be positive.")
        if total_iters <= 0:
            raise ValueError("Total iterations must be positive.")
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.eta_min = eta_min
        self.last_iter = last_iter
        self.restart = restart
        super(LinearWarmupCosineDecay, self).__init__(optimizer, last_iter)

    def load_state_dict(self, state_dict) -> None:
        """Restore the schedule from a checkpoint -- unless `restart` is set.

        torch's _LRScheduler.state_dict() dumps the scheduler's entire
        __dict__ and load_state_dict() blindly __dict__.update()s it back,
        so warmup_iters/total_iters/eta_min/base_lrs saved in the checkpoint
        silently override whatever the config asked for. That is exactly
        right when resuming an interrupted run, but wrong when *continuing a
        finished one*: there the cosine has already annealed to eta_min, and
        the point of relaunching is to run a new (usually lower-amplitude)
        cycle, which editing the config alone could never achieve.

        With restart=True the checkpoint's scheduler state is dropped
        entirely, so the schedule starts over at iteration 0 with the
        config's own warmup_iters/total_iters/eta_min and the optimizer's
        config lr as the new peak -- while model weights and optimizer
        (Adam moment) state still resume normally. warmup_iters/total_iters
        are then counted from the resume point, not from global_step 0.
        """
        if self.restart:
            return
        super().load_state_dict(state_dict)

    def get_lr(self) -> List[float]:
        # Compute the current iter
        current_iter = self.last_iter + 1
        self.last_iter = current_iter

        if current_iter < self.warmup_iters:
            # Linear warmup phase
            warmup_factor = (current_iter + 1) / self.warmup_iters
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay phase
            cosine_iter = current_iter - self.warmup_iters
            curr_progress = cosine_iter / max(
                1, self.total_iters - self.warmup_iters
            )
            cosine_decay_factor = 0.5 * (1 + math.cos(math.pi * curr_progress))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_decay_factor
                for base_lr in self.base_lrs
            ]
