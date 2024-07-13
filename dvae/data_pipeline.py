import os
from typing import Any, Callable, List

import lightning as lit
from datasets import load_dataset
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2

from .config import VaeConfig


class VaeDataset(Dataset):
    def __init__(self, images: List[Image.Image], transform: Callable) -> None:
        super().__init__()

        self.images = images
        self.transform = transform

    def __getitem__(self, index) -> Any:
        img = self.images[index]
        transformed_img = self.transform(img)

        return transformed_img


def get_train_dataloader(config: VaeConfig, images: List[Image.Image]):
    transform = v2.Compose(
        [
            v2.Resize(size=(config.h_params.img_height, config.h_params.img_width)),
            v2.RandomRotation(degrees=45),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(hue=0.3),
            v2.ToTensor(),
        ]
    )

    data = VaeDataset(images=images, transform=transform)
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
        os.makedirs(self.config.data.dir, exist_ok=True)
        data = load_dataset(self.config.data.hf_repo, token=self.hf_token)
        data.save_to_disk(self.config.data.dir)

    def setup(self, stage: str) -> None:
        dataset = load_dataset(
            self.config.data.hf_repo,
            token=self.hf_token,
            split="train",
        )

        self._images = [elt["images"] for elt in dataset]

    def train_dataloader(self) -> Any:
        return get_train_dataloader(self.config, images=self._images)
