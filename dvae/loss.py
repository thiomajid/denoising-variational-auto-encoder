import torch

from .config import VaeConfig


def loss_function(
    sample: torch.Tensor,
    reconstructed: torch.Tensor,
    config: VaeConfig,
    mu: torch.Tensor | None = None,
    logvar: torch.Tensor | None = None,
):
    loss_term = torch.nn.functional.mse_loss(
        reconstructed,
        sample,
    )

    if config.is_vae:
        kl_div = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - (logvar.exp() + config.optim.kl_eps)
        )

        loss_term = loss_term + config.optim.kl_beta * kl_div

    return loss_term
