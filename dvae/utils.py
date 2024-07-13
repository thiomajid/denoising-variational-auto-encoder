import math
from typing import Tuple

from .config import VaeConfig


def compute_last_conv_out_dim(config: VaeConfig):
    r"""
    Given that at each step out_dim is multipled by 2 => last out_dim = initial_out_dim * out_dim_scale^(n_block - 1).
    The exponent is n_block - 1 because for the first conv block out_dim is not scaled.

    Parameters
    ----------
    config : VaeConfig
        The configuration object holding the VAE hyper-parameters

    Returns
    -------
    int
        The dimension of the last convolution layer in a conv/conv_transpose nn.Sequential module
    """
    last_conv_out_dim = config.h_params.conv_out_dim * math.pow(
        config.h_params.conv_out_dim_scale,
        config.h_params.n_conv_block - 1,
    )

    return int(last_conv_out_dim)


def compute_conv_output_size(config: VaeConfig) -> Tuple[int, int, int]:
    """
    Compute the output size (channels, height, width) after n_conv_block convolutions

    Parameters
    ----------
    config : VaeConfig
        The configuration object holding the VAE hyper-parameters

    Returns
    -------
    Tuple[int, int, int]
        The number of channels, height, and width after convolutions
    """

    height = config.h_params.img_height
    width = config.h_params.img_width
    channels = config.h_params.conv_out_dim

    for _ in range(config.h_params.n_conv_block):
        height = (
            height + 2 * config.h_params.padding - config.h_params.kernel_size
        ) // config.h_params.stride + 1

        width = (
            width + 2 * config.h_params.padding - config.h_params.kernel_size
        ) // config.h_params.stride + 1

        channels *= config.h_params.conv_out_dim_scale

    return channels, height, width
