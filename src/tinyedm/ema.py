# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Taken from https://github.com/amorehead/NeMo/blob/main/nemo/collections/common/callbacks/ema.py with minor modifications.
import contextlib
import copy
import os
import threading
from typing import Any, Dict, Iterable, Tuple

import lightning as L
import torch
from lightning import Callback
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.utilities.exceptions import MisconfigurationException
import numpy as np


def sigma_rel_to_gamma(sigma_rel):
    t = sigma_rel**-2
    gamma = np.roots([1, 7, 16 - t, 12 - t]).real.max()
    return gamma


class EMA(Callback):
    """Implements Exponential Moving Averaging (EMA).

    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters with the prefix `ema`.

    Args:
        ema_length:  The “width” of its peak relative to training time.
        validate_original_weights: Validate the original weights, as apposed to the EMA weights.
        every_n_steps: Apply EMA every N steps.
        cpu_offload: Offload weights to CPU.
    """

    def __init__(
        self,
        ema_length: float,
        validate_original_weights: bool = False,
        every_n_steps: int = 1,
        cpu_offload: bool = False,
    ):
        if not (0 <= ema_length <= 0.2886):
            raise MisconfigurationException(
                "EMA length value must be between 0 and 0.2886"
            )
        self.ema_length = ema_length
        self.gamma = sigma_rel_to_gamma(ema_length)
        self.validate_original_weights = validate_original_weights
        self.every_n_steps = every_n_steps
        self.cpu_offload = cpu_offload

    def on_fit_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        device = pl_module.device if not self.cpu_offload else torch.device("cpu")
        trainer.optimizers = [
            EMAOptimizer(
                optim,
                device=device,
                gamma=self.gamma,
                every_n_steps=self.every_n_steps,
                current_step=trainer.global_step,
            )
            for optim in trainer.optimizers
            if not isinstance(optim, EMAOptimizer)
        ]

    def on_validation_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_validation_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def _should_validate_ema_weights(self, trainer: "L.Trainer") -> bool:
        return not self.validate_original_weights and self._ema_initialized(trainer)

    def _ema_initialized(self, trainer: "L.Trainer") -> bool:
        """Check if EMA is initialized.

        Args:
            trainer: a Trainer
        Returns:
            True if EMA is initialized, False otherwise
        """
        return any(
            isinstance(optimizer, EMAOptimizer) for optimizer in trainer.optimizers
        )

    def swap_model_weights(self, trainer: "L.Trainer"):
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, EMAOptimizer)
            optimizer.switch_main_parameter_weights()


@torch.no_grad()
def ema_update(
    ema_model_tuple: Tuple[Any], current_model_tuple: Tuple[Any], decay: float
):
    """Exponential moving average update.

    Args:
        ema_model_tuple: a tuple of EMA model parameters
        current_model_tuple: a tuple of current model parameters
        decay: the EMA decay factor
    """
    torch._foreach_mul_(ema_model_tuple, decay)
    torch._foreach_add_(
        ema_model_tuple,
        current_model_tuple,
        alpha=(1.0 - decay),
    )


def run_ema_update_cpu(
    ema_model_tuple, current_model_tuple, decay, pre_sync_stream=None
):
    """Run EMA updates on the CPU.

    Args:
        ema_model_tuple: a tuple of EMA model parameters
        current_model_tuple: a tuple of current model parameters
        decay: the EMA decay factor
        pre_sync_stream: a stream to synchronize on before running the update
    """
    if pre_sync_stream is not None:
        pre_sync_stream.synchronize()

    ema_update(ema_model_tuple, current_model_tuple, decay)


class EMAOptimizer(torch.optim.Optimizer):
    r"""EMAOptimizer is a wrapper for torch.optim.Optimizer that computes Exponential Moving Average
    of parameters registered in the optimizer.

    EMA parameters are automatically updated after every step of the optimizer
    with the following formula:

        ema_weight = decay * ema_weight + (1 - decay) * training_weight

    To access EMA parameters, use ``swap_ema_weights()`` context manager to
    perform a temporary in-place swap of regular parameters with EMA
    parameters.

    Notes:
        - EMAOptimizer is not compatible with APEX AMP O2.

    Args:
        optimizer (torch.optim.Optimizer): optimizer to wrap
        device (torch.device): device for EMA parameters
        decay (float): decay factor

    Returns:
        An instance of torch.optim.Optimizer that computes EMA of
        parameters

    Example:
        model = Model().to(device)
        opt = torch.optim.Adam(model.parameters())

        opt = EMAOptimizer(opt, device, 0.9999)

        for epoch in range(epochs):
            training_loop(model, opt)

            regular_eval_accuracy = evaluate(model)

            with opt.swap_ema_weights():
                ema_eval_accuracy = evaluate(model)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        gamma: float,
        every_n_steps: int = 1,
        current_step: int = 0,
    ):
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device
        self.current_step = current_step
        self.every_n_steps = every_n_steps

        self.first_iteration = True
        self.rebuild_ema_params = True
        self.stream = None
        self.thread = None

        self.ema_params = ()

    def all_parameters(self) -> Iterable[torch.Tensor]:
        """Returns an iterable of all parameters in the optimizer.

        Returns:
            an iterable of all parameters in the optimizer
        """
        return (param for group in self.param_groups for param in group["params"])

    def step(self, closure=None, **kwargs):
        """Takes a closure step that reevaluates the model and returns the loss.

        Args:
            closure: a closure
            kwargs: additional keyword arguments
        Returns:
            the loss
        """
        self.synchronize()

        if self.first_iteration:
            if any(p.is_cuda for p in self.all_parameters()):
                self.stream = torch.cuda.Stream()

            self.first_iteration = False

        if self.rebuild_ema_params:
            opt_params = list(self.all_parameters())

            self.ema_params += tuple(
                copy.deepcopy(param.data.detach()).to(self.device)
                for param in opt_params[len(self.ema_params) :]
            )
            self.rebuild_ema_params = False

        loss = self.optimizer.step(closure)

        if self._should_update_at_step():
            self.update()
        self.current_step += 1
        return loss

    def _should_update_at_step(self) -> bool:
        """Check if we should update at the current step.

        Returns:
            True if we should update at the current step, False otherwise
        """
        return self.current_step % self.every_n_steps == 0

    @torch.no_grad()
    def update(self):
        """Updates EMA parameters."""
        decay = (1 - 1 / (self.current_step + 1)) ** (self.gamma + 1)
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.stream):
            current_model_state = tuple(
                param.data.to(self.device, non_blocking=True)
                for param in self.all_parameters()
            )

            if self.device.type == "cuda":
                ema_update(self.ema_params, current_model_state, decay)

        if self.device.type == "cpu":
            self.thread = threading.Thread(
                target=run_ema_update_cpu,
                args=(
                    self.ema_params,
                    current_model_state,
                    decay,
                    self.stream,
                ),
            )
            self.thread.start()

    def swap_tensors(self, tensor1, tensor2):
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)

    def switch_main_parameter_weights(self):
        self.synchronize()
        for param, ema_param in zip(self.all_parameters(), self.ema_params):
            self.swap_tensors(param.data, ema_param)

    @contextlib.contextmanager
    def swap_ema_weights(self, enabled: bool = True):
        r"""A context manager to in-place swap regular parameters with EMA parameters. It swaps back
        to the original regular parameters on context manager exit.

        Args:
            enabled (bool): whether the swap should be performed
        """

        if enabled:
            self.switch_main_parameter_weights()
        try:
            yield
        finally:
            if enabled:
                self.switch_main_parameter_weights()

    def __getattr__(self, name):
        """Proxy non-overridden attributes to the underlying optimizer."""
        return getattr(self.optimizer, name)

    def synchronize(self):
        if self.stream is not None:
            self.stream.synchronize()

        if self.thread is not None:
            self.thread.join()

    def state_dict(self):
        self.synchronize()

        state_dict = {
            "opt": self.optimizer.state_dict(),
            "ema": self.ema_params,
            "current_step": self.current_step,
            "gamma": self.gamma,
            "every_n_steps": self.every_n_steps,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.synchronize()

        self.optimizer.load_state_dict(state_dict["opt"])
        self.ema_params = tuple(
            param.to(self.device) for param in copy.deepcopy(state_dict["ema"])
        )
        self.current_step = state_dict["current_step"]
        self.gamma = state_dict["gamma"]
        self.every_n_steps = state_dict["every_n_steps"]
        self.rebuild_ema_params = False
