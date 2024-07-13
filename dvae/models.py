import math
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .config import VaeConfig
from .utils import compute_conv_output_size, compute_last_conv_out_dim

_ActivationFn = Callable[[torch.Tensor], torch.Tensor]


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        kernel_size: int,
        padding: int,
        stride: int,
        activation: _ActivationFn,
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
        activation: _ActivationFn,
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


class VaeEncoder(nn.Module):
    def __init__(self, config: VaeConfig, activation: _ActivationFn) -> None:
        super().__init__()

        self.activation = activation
        self.config = config

        self.conv_blocks = self.__init_conv_blocks()

        out_dim_scale = self.config.h_params.conv_out_dim_scale
        last_conv_out_dim = compute_last_conv_out_dim(self.config)
        _, img_height, img_width = compute_conv_output_size(self.config)

        self.fc = nn.Linear(
            last_conv_out_dim * img_height * img_width,
            last_conv_out_dim * out_dim_scale,
        )

        self.fc_mu = nn.Linear(
            last_conv_out_dim * out_dim_scale,
            self.config.h_params.latent_dim,
        )

        self.fc_logvar = nn.Linear(
            last_conv_out_dim * out_dim_scale,
            self.config.h_params.latent_dim,
        )

    def __init_conv_blocks(self):
        """
        Creates a nn.Sequential module containing config.h_params.n_conv_blocks
        convolution layers for the encoder

        Returns
        -------
        nn.ModuleList
            A nn.ModuleList module containing the encoder's convolution blocks
        """
        conv_modules = nn.ModuleList()
        in_channels = self.config.h_params.input_dim
        out_channels = self.config.h_params.conv_out_dim
        n_blocks = self.config.h_params.n_conv_block
        out_dim_scale = self.config.h_params.conv_out_dim_scale

        for _ in range(n_blocks):
            block = ConvBlock(
                input_dim=in_channels,
                out_dim=out_channels,
                kernel_size=self.config.h_params.kernel_size,
                padding=self.config.h_params.padding,
                stride=self.config.h_params.stride,
                activation=self.activation,
            )

            conv_modules.append(block)
            in_channels = out_channels
            out_channels = in_channels * out_dim_scale

        return conv_modules

    def forward(self, x: torch.Tensor):
        h = x
        for block in self.conv_blocks:
            h = block(h)

        # [batch_size, n_channels, height, width] => [batch_size, n_features]
        h = h.view(h.size(0), -1)
        h = self.activation(self.fc(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class VaeDecoder(nn.Module):
    def __init__(self, config: VaeConfig, activation: _ActivationFn) -> None:
        super().__init__()

        self.config = config
        self.activation = activation

        out_dim_scale = self.config.h_params.conv_out_dim_scale
        self._last_conv_out_dim = compute_last_conv_out_dim(self.config)
        _, self._conv_img_height, self._conv_img_width = compute_conv_output_size(
            self.config
        )

        self.fc_from_latent = nn.Linear(
            config.h_params.latent_dim,
            self._last_conv_out_dim * out_dim_scale,
        )
        self.fc_to_conv_transpose = nn.Linear(
            self._last_conv_out_dim * out_dim_scale,
            self._last_conv_out_dim * self._conv_img_width * self._conv_img_height,
        )

        self.conv_transpose_blocks = self.__init_conv_transpose_blocks()

    def __init_conv_transpose_blocks(self):
        out_dim_scale = self.config.h_params.conv_out_dim_scale
        last_conv_out_dim = self.config.h_params.conv_out_dim * math.pow(
            out_dim_scale,
            self.config.h_params.n_conv_block - 1,
        )

        modules = nn.ModuleList()
        in_channels = last_conv_out_dim
        out_channels = in_channels // out_dim_scale
        n_blocks = self.config.h_params.n_conv_block

        for _ in range(n_blocks):
            block = ConvTransposeBlock(
                input_dim=in_channels,
                out_dim=out_channels,
                kernel_size=self.config.h_params.kernel_size,
                padding=self.config.h_params.padding,
                stride=self.config.h_params.stride,
                activation=self.activation,
            )

            modules.append(block)
            in_channels = out_channels
            out_channels = in_channels // out_dim_scale

        return modules

    def forward(self, z: torch.Tensor):
        h = self.activation(self.fc_from_latent(z))
        h = self.activation(self.fc_to_conv_transpose(h))

        # (batch_size, n_features) => (batch_size, n_channels, height, width)
        h = h.view(
            -1, self._last_conv_out_dim, self._conv_img_height, self._conv_img_width
        )

        for block in self.conv_transpose_blocks:
            h = block(h)

        return torch.sigmoid(h)


class DenoisingVAE(nn.Module):
    def __init__(self, config: VaeConfig) -> None:
        super().__init__()

        activation = None
        match config.activation_fn:
            case "gelu":
                activation = F.gelu
            case "leaky_relu":
                activation = F.leaky_relu
            case "silu":
                activation = F.silu
            case "relu":
                activation = F.relu
            case _:
                raise ValueError(
                    f"{config.activation_fn} is not a known activation function"
                )

        self.activation = activation
        self.config = config

        self.encoder = VaeEncoder(config, self.activation)
        self.decoder = VaeDecoder(config, self.activation)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    @torch.no_grad()
    def generate(self, num_samples: int) -> torch.Tensor:
        z = torch.randn((num_samples, self.config.h_params.latent_dim))
        return self.decode(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        reconstructed = self.decode(z)

        return reconstructed, mu, logvar
