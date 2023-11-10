import hydra
from omegaconf import DictConfig, OmegaConf
import logging

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    """ Main entry point for the software. Essentially able to deligate execution to
        - Training module
        - Inference module
        - Evaluation module
        - Anything else?

        Uses Hydra to manage configuration and logging. Hydra's config is located
        in the conf/config.yaml file, which is the default config. Add new values there
        and access them in code with cfg.<value_name>
        e.g. cfg.training.model

        Run param sweeps
    """
    # save the cfg to a global variable so it can be accessed anywhere
    global config
    config = cfg
    log.info(OmegaConf.to_yaml(cfg))



if __name__ == "__main__":
    main()