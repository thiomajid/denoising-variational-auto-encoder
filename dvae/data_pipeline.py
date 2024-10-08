import os
from typing import Any, Callable, List

import lightning as lit
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import v2

from .config import VaeConfig


class VaeDataset(Dataset):
    def __init__(
        self, images: List[Image.Image], device: str, transform: Callable
    ) -> None:
        super().__init__()

        self.images = images
        self.transform = transform
        self.device = device

    def __getitem__(self, index) -> Any:
        img = self.images[index]
        transformed_img: torch.Tensor = self.transform(img)

        return transformed_img

    def __len__(self):
        return len(self.images)


def get_train_dataloader(config: VaeConfig, data: Dataset):
    loader = DataLoader(
        data,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        persistent_workers=True,
        shuffle=True,
    )

    return loader


class VaeDataModule(lit.LightningDataModule):
    def __init__(self, config: VaeConfig, hf_token: str) -> None:
        super().__init__()

        self.config = config
        self.hf_token = hf_token

    def prepare_data(self) -> None:
        if self.config.data.source == "hf":
            os.makedirs(self.config.data.dir, exist_ok=True)
            data = load_dataset(self.config.data.hf_repo, token=self.hf_token)
            data.save_to_disk(self.config.data.dir)
        else:
            MNIST(
                root=self.config.data.dir,
                download=True,
                transform=v2.ToTensor(),
            )

    def setup(self, stage: str) -> None:
        self.download_train_data()

    def download_train_data(self):
        if self.config.data.source == "mnist":
            self.data = MNIST(
                root=self.config.data.dir,
                download=True,
                transform=v2.ToTensor(),
            )
            return self.data

        dataset = load_dataset(
            self.config.data.hf_repo,
            token=self.hf_token,
            split="train",
        )

        self._images: List[Image.Image] = [elt["image"] for elt in dataset]
        transform = v2.Compose(
            [
                v2.Resize(
                    size=(
                        self.config.img_height,
                        self.config.img_width,
                    )
                ),
                v2.RandomRotation(degrees=45),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(hue=0.3, brightness=0.25),
                v2.ToTensor(),
            ]
        )

        self.data = VaeDataset(
            images=self._images,
            transform=transform,
            device=self.config.optim.device,
        )
        return self.data

    def train_dataloader(self) -> Any:
        return get_train_dataloader(self.config, data=self.download_train_data())
