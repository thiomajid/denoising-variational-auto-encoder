from dataclasses import dataclass


@dataclass(frozen=True)
class _VaeHparams:
    """
    _summary_


    """

    input_dim: int
    conv_out_dim: int
    conv_out_dim_scale: 2
    latent_dim: int
    kernel_size: int
    stride: int
    padding: int
    n_conv_block: int
    img_width: int
    img_height: int


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
    activation_fn: str
    data: _VaeDataConfig
    h_params: _VaeHparams
