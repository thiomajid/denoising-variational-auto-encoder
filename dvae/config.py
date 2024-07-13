from dataclasses import dataclass
from typing import Literal

_ActivationFn = Literal["gelu", "relu", "leaky_relu", "silu"]


@dataclass(frozen=True)
class _VaeHparams:
    """
    _summary_


    """

    input_dim: int
    latent_dim: int
    kernel_size: int
    stride: int
    padding: int
    n_conv_block: int
    img_dim: int


@dataclass(frozen=True)
class _VaeDataConfig:
    """
    _summary_


    """

    dir: str
    num_workers: int
    batch_size: int


@dataclass(frozen=True)
class VaeConfig:
    """
    _summary_


    """

    model_name: str
    epochs: int
    lr: float
    activation_fn: _ActivationFn
    data: _VaeDataConfig
    h_params: _VaeHparams
