import torch


def loss_function(
    sample: torch.Tensor,
    reconstructed: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
):
    bce_loss = torch.nn.functional.binary_cross_entropy(
        reconstructed,
        sample,
        reduction="sum",
    )
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return bce_loss + kl_div
