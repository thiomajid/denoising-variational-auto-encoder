from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .config import VaeConfig


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        kernel_size: int,
        padding: int,
        stride: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self.activation = activation
        self.conv = nn.Conv2d(
            in_channels=input_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

    def forward(self, hidden_state: torch.Tensor):
        return self.activation(self.conv(hidden_state))


class ConvTransposeBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        kernel_size: int,
        padding: int,
        stride: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self.activation = activation
        self.conv = nn.ConvTranspose2d(
            in_channels=input_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

    def forward(self, hidden_state: torch.Tensor):
        return self.activation(self.conv(hidden_state))


class Encoder(nn.Module):
    def __init__(self, config: VaeConfig) -> None:
        super().__init__()

        activation = None

        self.config = config
        match config.activation_fn:
            case "gelu":
                activation = F.gelu
            case "leaky_relu":
                activation = F.leaky_relu
            case "silu":
                activation = F.silu
            case _:
                activation = F.relu

        self.activation = activation

    def forward(self, x: torch.Tensor):
        pass


class Decoder(nn.Module):
    def __init__(self, config: VaeConfig) -> None:
        super().__init__()

        pass

    def forward(self, x: torch.Tensor):
        pass


class DenoisingVAE(nn.Module):
    def __init__(self, config: VaeConfig) -> None:
        super().__init__()

        pass

    def forward(self, x: torch.Tensor):
        pass
