import hydra
import torch
from hydra.core.config_store import ConfigStore

from dvae.config import VaeConfig
from dvae.models import DenoisingVAE

config_store = ConfigStore.instance()
config_store.store(name="dvae_config", node=VaeConfig)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: VaeConfig):
    vae = DenoisingVAE(config)
    print(vae)
    out, _, _ = vae(torch.randn((4, 3, 512, 512)))
    print(out.shape)

    # logger = TensorBoardLogger(save_dir=config.ckpt_dir, name="dvae-v0")
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
    # trainer = lit.Trainer(
    #     accelerator=config.accelerator,
    #     devices=config.num_device,
    #     max_epochs=config.epochs,
    #     precision=config.precision,
    #     logger=logger,
    #     callbacks=callbacks,
    #     enable_checkpointing=True,
    #     enable_model_summary=True,
    #     enable_progress_bar=True,
    # )


if __name__ == "__main__":
    main()
