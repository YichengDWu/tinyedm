import pytest
import torch
from tinyedm.metric import WeightedMeanSquaredError, _weighted_sum_squared_error_update


def test_weighted_mean_squared_error_metric():
    metric = WeightedMeanSquaredError()

    weights = torch.randn(8).exp()
    preds = torch.randn((8, 3, 32, 32))
    target = torch.randn((8, 3, 32, 32))

    # Update state
    metric(weights, preds, target)
    computed_mse = metric.compute()

    # Check if the state is updated correctly
    assert torch.allclose(
        computed_mse, torch.mean(weights.view(-1, 1, 1, 1) * (preds - target) ** 2)
    )
    assert metric.total == 8
