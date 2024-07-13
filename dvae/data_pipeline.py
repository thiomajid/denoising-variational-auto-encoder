import os
from typing import Any, Callable, Tuple

import lightning as lit
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import v2

from .config import VaeConfig


class VaeDataset(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Callable[[str], bool] | None = None,
    ):
        super().__init__(root, transform, target_transform, loader, is_valid_file)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index)


def get_train_dataloader(config: VaeConfig):
    transform = v2.Compose(
        [
            v2.Resize(size=(config.h_params.img_height, config.h_params.img_width)),
            v2.RandomRotation(degrees=45),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(hue=0.3),
            v2.ToTensor(),
        ]
    )

    data = VaeDataset(root=config.data.dir, transform=transform)
    loader = DataLoader(
        data,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=True,
    )

    return loader


class VaeDataModule(lit.LightningDataModule):
    def __init__(self, config: VaeConfig, hf_token: str) -> None:
        super().__init__()

        self.config = config
        self.hf_token = hf_token

    def prepare_data(self) -> None:
        load_dataset(self.config.data.hf_repo, token=self.hf_token)

    def setup(self, stage: str) -> None:
        data = load_dataset(self.config.data.hf_repo, token=self.hf_token)

        if stage == "train":
            os.makedirs(self.config.data.dir, exist_ok=True)
            data["train"].save_to_disk(self.config.data.dir)

    def train_dataloader(self) -> Any:
        return get_train_dataloader(self.config)
