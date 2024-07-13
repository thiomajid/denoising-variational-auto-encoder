import hydra
from hydra.core.config_store import ConfigStore

from dvae.config import VaeConfig
from dvae.models import DenoisingVAE

config_store = ConfigStore.instance()
config_store.store(name="dvae_config", node=VaeConfig)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: VaeConfig):
    vae = DenoisingVAE(config)
    print(vae)


if __name__ == "__main__":
    main()
