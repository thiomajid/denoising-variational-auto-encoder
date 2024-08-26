from typing import Tuple

import lightning as lit
import torch
from torch import nn

from .config import VaeConfig
from .loss import loss_function
from .utils import (
    compute_conv_output_size,
    compute_last_conv_out_dim,
    get_activation_function,
    plot_generated_images,
)


class ConvBlock(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, config: VaeConfig) -> None:
        super().__init__()

        self.config = config
        self.activation = get_activation_function(config.activation_fn)

        self.conv = nn.Conv2d(
            in_channels=input_dim,
            out_channels=out_dim,
            kernel_size=config.h_params.kernel_size,
            padding=config.h_params.padding,
            stride=config.h_params.stride,
            bias=self.config.h_params.conv_bias,
        )

        # Xavier initialization
        torch.nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            torch.nn.init.zeros_(self.conv.bias)

        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, hidden_state: torch.Tensor):
        h = self.conv(hidden_state)
        if self.config.h_params.normalize:
            h = self.norm(h)

        return self.activation(h)


class ConvTransposeBlock(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, config: VaeConfig) -> None:
        super().__init__()

        self.config = config
        self.activation = get_activation_function(config.activation_fn)

        self.conv = nn.ConvTranspose2d(
            in_channels=input_dim,
            out_channels=out_dim,
            kernel_size=config.h_params.kernel_size,
            padding=config.h_params.padding,
            stride=config.h_params.stride,
            bias=self.config.h_params.conv_bias,
        )
        # Xavier initialization
        torch.nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            torch.nn.init.zeros_(self.conv.bias)

        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, hidden_state: torch.Tensor):
        h = self.conv(hidden_state)
        if self.config.h_params.normalize:
            h = self.norm(h)

        return self.activation(h)


class VaeEncoder(nn.Module):
    def __init__(self, config: VaeConfig) -> None:
        super().__init__()

        self.config = config
        self.activation = get_activation_function(config.activation_fn)

        self.conv_blocks = self.__init_conv_blocks()

        out_dim_scale = self.config.h_params.conv_out_dim_scale
        last_conv_out_dim = compute_last_conv_out_dim(self.config)
        _, img_height, img_width = compute_conv_output_size(self.config)

        self.fc = (
            nn.Linear(
                last_conv_out_dim * img_height * img_width,
                last_conv_out_dim * out_dim_scale,
            )
            if self.config.is_vae
            else nn.Linear(
                last_conv_out_dim * img_height * img_width,
                self.config.h_params.latent_dim,
            )
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
                config=self.config,
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
        h: torch.Tensor = self.activation(self.fc(h))

        if not self.config.is_vae:
            return h

        mu: torch.Tensor = self.fc_mu(h)
        logvar: torch.Tensor = self.fc_logvar(h)

        return mu, logvar


class VaeDecoder(nn.Module):
    def __init__(self, config: VaeConfig) -> None:
        super().__init__()

        self.config = config
        self.activation = get_activation_function(config.activation_fn)

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
        self.output = nn.Sequential(
            nn.Upsample(
                size=(config.img_height, config.img_width),
                mode="bilinear",
                align_corners=False,
            ),
            nn.Sigmoid(),
        )

    def __init_conv_transpose_blocks(self):
        out_dim_scale = self.config.h_params.conv_out_dim_scale
        last_conv_out_dim = compute_last_conv_out_dim(self.config)

        modules = nn.ModuleList()
        in_channels = last_conv_out_dim
        out_channels = in_channels // out_dim_scale
        n_blocks = self.config.h_params.n_conv_block

        # up to n_blocks - 1 because the last conv layer must produce 3 channels
        for _ in range(n_blocks - 1):
            block = ConvTransposeBlock(
                input_dim=in_channels,
                out_dim=out_channels,
                config=self.config,
            )

            modules.append(block)
            in_channels = out_channels
            out_channels = in_channels // out_dim_scale

        # registering the last conv transpose module
        modules.append(
            ConvTransposeBlock(
                input_dim=in_channels,
                out_dim=self.config.h_params.input_dim,
                config=self.config,
            )
        )

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

        if (
            h.shape[-2] != self.config.img_width
            and h.shape[-1] != self.config.img_height
        ):
            h = self.output(h)

        return h


class DenoisingVAE(nn.Module):
    def __init__(self, config: VaeConfig) -> None:
        super().__init__()

        self.config = config
        self.encoder = VaeEncoder(config)
        self.decoder = VaeDecoder(config)

    def auto_encoder_forward(self, x: torch.Tensor):
        h = self.encoder(x)
        reconstructed = self.decoder(h)

        return reconstructed

    def vae_forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        reconstructed = self.decoder(z)

        return reconstructed, mu, logvar

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x: torch.Tensor):
        if self.config.is_vae:
            return self.vae_forward(x)

        return self.auto_encoder_forward(x)

    @torch.no_grad()
    def generate(self, num_samples: int) -> torch.Tensor:
        """
        Passes gaussian noise through the decoder to generate num_samples images

        Parameters
        ----------
        num_samples : int
            The number of images to generate

        Returns
        -------
        torch.Tensor
            A tensor containing logits that should be normalized using the sigmoid function
        """
        z = torch.randn(
            (num_samples, self.config.h_params.latent_dim),
            device=self.config.optim.device,
        )

        return torch.nn.functional.sigmoid(self.decode(z))


class LitDenoisingVAE(lit.LightningModule):
    def __init__(self, config: VaeConfig, hf_token: str) -> None:
        super().__init__()

        self.config = config
        self.hf_token = hf_token
        self.model = DenoisingVAE(config)

        self.example_input_array = torch.randn(
            (
                1,
                config.h_params.input_dim,
                config.img_height,
                config.img_width,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Any:
        return self.model(x)

    def generate(self, num_samples: int):
        return self.model.generate(num_samples)

    def training_step(self, batch: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]):
        if isinstance(batch, tuple):
            img, _ = batch
        else:
            img = batch

        img = img.to(self.device)

        loss = None
        if self.config.is_vae:
            reconstructed, mu, logvar = self.model(img)
            loss = loss_function(
                sample=batch,
                reconstructed=reconstructed,
                mu=mu,
                logvar=logvar,
                config=self.config,
            )
        else:
            reconstructed = self.model(img)
            loss = loss_function(
                sample=batch,
                reconstructed=reconstructed,
                config=self.config,
            )

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.optim.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.optim.t_max,
            eta_min=self.config.optim.eta_min,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def on_train_epoch_end(self) -> None:
        n_samples = 4
        generated = self.model.generate(num_samples=n_samples)
        plot_generated_images(generated)
