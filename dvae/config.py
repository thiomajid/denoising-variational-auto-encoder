from dataclasses import dataclass


@dataclass(frozen=True)
class _VaeHparams:
    """
    Hyper-parameters used to define the model architecture
    """

    input_dim: int
    conv_out_dim: int
    conv_out_dim_scale: int
    latent_dim: int
    kernel_size: int
    stride: int
    padding: int
    n_conv_block: int
    normalize: bool


@dataclass(frozen=True)
class _VaeDataConfig:
    """
    Config class for everything related to the training data pipeline.
    """

    dir: str
    num_workers: int
    batch_size: int
    hf_repo: str
    source: str


@dataclass(frozen=True)
class _VaeOptimConfig:
    device: str
    accelerator: str
    num_device: int
    precision: str
    epochs: int
    lr: float
    eta_min: float
    t_max: int
    kl_eps: float
    kl_beta: float


@dataclass(frozen=True)
class VaeConfig:
    """
    Config class used to define VAE hyper-parameters and training data pipeline.
    """

    model_name: str
    model_hf_repo: str
    img_width: int
    img_height: int
    activation_fn: str
    ckpt_dir: str
    optim: _VaeOptimConfig
    data: _VaeDataConfig
    h_params: _VaeHparams
