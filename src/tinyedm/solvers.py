import torch


class DeterministicSolver:
    """
    A deterministic solver for diffusion models. Algorithm 1 in [1] with `sigma(t)=t` and
    `s(t)=1`.

    References:
        [1] Karras T, Aittala M, Aila T, et al. Elucidating the design space of diffusion-based generative models[J]. Advances in Neural Information Processing Systems, 2022, 35: 26565-26577.
    """

    def __init__(
        self,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        dtype: str | None = None,
    ):
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        
        if dtype is None or dtype == "float32":
            self.dtype = torch.float32
        elif dtype == "float16":
            self.dtype = torch.float16
        elif dtype == "float64":
            self.dtype = torch.float64

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
            denoised = model(x0, t0.to(device=x0.device), class_labels).to(self.dtype)
            dx = (x0 - denoised) / t0
            x1 = x0 + (t1 - t0) * dx

            if i < self.num_steps - 1:
                denoised_prime = model(x1, t1.to(device=x0.device), class_labels).to(
                    self.dtype
                )
                dx_prime = (x1 - denoised_prime) / t1
                x1 = x0 + (t1 - t0) * (0.5 * dx + 0.5 * dx_prime)

        return x1.to(x0.dtype)
