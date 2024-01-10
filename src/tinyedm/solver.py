import torch
from typing import Protocol


class DiffusionSolver(Protocol):
    def solve(self, model, x0):
        pass


class DeterministicSolver:
    def __init__(
        self,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        dtype: torch.dtype = torch.float64,
    ):
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.dtype = dtype

        step_indices = torch.arange(num_steps, dtype=dtype)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        self.t_steps = torch.cat([t_steps, torch.zeros(1)])  # t_N = 0

    def solve(self, model, x0):
        x1 = x0.to(self.dtype) * self.t_steps[0]
        for i, (t0, t1) in enumerate(zip(self.t_steps[:-1], self.t_steps[1:])):
            x0 = x1
            denoised = model(x0.to(model.dtype), t0.to(model.dtype)).to(
                self.dtype
            )  # mixed precision
            dx = (x0 - denoised) / t0
            x1 = x0 + (t1 - t0) * dx

            if i < self.num_steps - 1:
                denoised = model(x0.to(model.dtype), t0.to(model.dtype)).to(self.dtype)
                dx_prime = (x1 - denoised) / t1
                x1 = x0 + (t1 - t0) * (0.5 * dx + 0.5 * dx_prime)

        return x1.to(x0.dtype)
