import torch


class DeterministicSolver:
    """
    A deterministic solver for diffusion models.

    Args:
        num_steps: The number of steps to take.
        sigma_min: The minimum value of sigma.
        sigma_max: The maximum value of sigma.
        rho: The value of rho.
        dtype: The dtype of the solver.

    Methods:
        solve: Solve the diffusion model.
            Args:
                model: The diffusion model.
                x0: The initial value. Assumed to be **standard normal**.

    """

    def __init__(
        self,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.dtype = dtype  # The dtype of the solution

        step_indices = torch.arange(num_steps, dtype=dtype)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        self.t_steps = torch.cat([t_steps, torch.zeros(1)]).to(dtype)

    def solve(self, model, x0, class_labels=None):
        x0 = x0.to(self.dtype)
        x1 = x0 * self.t_steps[0]
        for i, (t0, t1) in enumerate(zip(self.t_steps[:-1], self.t_steps[1:])):
            x0 = x1
            denoised = model(x0, t0, class_labels).to(self.dtype)
            dx = (x0 - denoised) / t0
            x1 = x0 + (t1 - t0) * dx

            if i < self.num_steps - 1:
                denoised_prime = model(x1.to, t1, class_labels).to(self.dtype)
                dx_prime = (x1 - denoised_prime) / t1
                x1 = x0 + (t1 - t0) * (0.5 * dx + 0.5 * dx_prime)

        return x1.to(x0.dtype)
