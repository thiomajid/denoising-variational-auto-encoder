import torch

from .config import VaeConfig


def loss_function(
    sample: torch.Tensor,
    reconstructed: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    config: VaeConfig,
):
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        reconstructed,
        sample,
        reduction="sum",
    )
    kl_div = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - (logvar.exp() + config.optim.kl_eps)
    )

    total = bce_loss + config.optim.kl_beta * kl_div

    if torch.isnan(total):
        print(f"NaN detected - BCE: {bce_loss}, KL: {kl_div}")
        print(f"mu range: {mu.min()} to {mu.max()}")
        print(f"logvar range: {logvar.min()} to {logvar.max()}")

    return total
