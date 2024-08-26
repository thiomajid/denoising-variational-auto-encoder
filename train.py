import hydra

# import lightning as lit
import torch
from hydra.core.config_store import ConfigStore

# from lightning.pytorch.callbacks import (
#     LearningRateFinder,
#     LearningRateMonitor,
#     ModelCheckpoint,
# )
# from dvae.data_pipeline import VaeDataModule
from dvae.config import VaeConfig
from dvae.models import DenoisingVAE

config_store = ConfigStore.instance()
config_store.store(name="dvae_config", node=VaeConfig)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: VaeConfig):
    vae = DenoisingVAE(config)
    print(vae)
    out = None
    sample = torch.randn(
        (1, config.h_params.input_dim, config.img_height, config.img_width)
    )
    if config.is_vae:
        out, _, _ = vae(sample)
    else:
        out = vae(sample)

    print(out.shape)

    # callbacks = [
    #     LearningRateFinder(),
    #     LearningRateMonitor(logging_interval="step", log_weight_decay=True),
    #     ModelCheckpoint(
    #         dirpath=config.ckpt_dir,
    #         filename="dvae-v0",
    #         save_top_k=1,
    #         verbose=True,
    #         monitor="train_loss",
    #         mode="min",
    #         save_weights_only=False,
    #     ),
    # ]

    # datamodule = VaeDataModule(config=config, hf_token="")

    # trainer = lit.Trainer(
    #     accelerator=config.accelerator,
    #     devices=config.num_device,
    #     max_epochs=config.epochs,
    #     callbacks=callbacks,
    #     enable_checkpointing=True,
    #     enable_model_summary=True,
    #     enable_progress_bar=True,
    # )

    # trainer.fit(vae, train_dataloaders=datamodule.train_dataloader())


if __name__ == "__main__":
    main()
