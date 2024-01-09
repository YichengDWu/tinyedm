from torchmetrics.metric import Metric
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torch import Tensor, tensor
import torch
from typing import Sequence

def _weighted_sum_squared_error_update(weights: Tensor, preds: Tensor, target: Tensor) -> tuple[Tensor, int]:
    N = target.shape[0]
    preds = preds.view(N, -1)
    target = target.view(N, -1)
    weights = weights.view(N, 1)
    
    diff = preds - target
    weighted_sum_squared_error = torch.mean(weights * diff * diff, dim=1)
    return weighted_sum_squared_error.sum(), N

class WeightedMeanSquaredError(Metric):

    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    plot_lower_bound: float = 0.0

    weighted_sum_squared_error: Tensor
    total: Tensor

    def __init__(
        self, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("weighted_sum_squared_error", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, weight: Tensor, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        weighted_sum_squared_error, num_obs = _weighted_sum_squared_error_update(weight, preds, target)

        self.weighted_sum_squared_error += weighted_sum_squared_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean squared error over state."""
        return self.weighted_sum_squared_error / self.total

    def plot(
        self, val: Tensor | Sequence[Tensor] | None=None, ax: _AX_TYPE | None = None
    ) -> _PLOT_OUT_TYPE:
        return self._plot(val, ax)