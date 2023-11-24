import logging
import os
import wandb
import src.utils.utilities as utils
from src.ml.summarizer import Summarizer
import src.datasets.xSum as xSum

from huggingface_hub.hf_api import HfFolder


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def setupEnvVariables(args):
    os.environ["WANDB_API_KEY"] = "3e554d2dafac7c22b4e348afff4b69a9e6d49e81"
    os.environ["WANDB_PROJECT"] = "adv_nlp2023"
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # 'checkpoint' or
    os.environ["HF_TOKEN"] = "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"
    HfFolder.save_token(os.environ["HF_TOKEN"])


def initWandb(config=None):
    if config is None:
        return wandb.init(project="adv_nlp2023")
    else:
        return wandb.init(project="adv_nlp2023", config=config)


def main():
    """Main entry point for the software. Essentially able to deligate execution to
    - Training module
    - Inference module
    - Evaluation module
    - Anything else?

    Use Weights & Biases to manage parameter sweeps. Hydra's config is located
    in the conf/config.yaml file, which is the default config. Add new values there
    and access them in code with cfg.<value_name>
    e.g. cfg.training.model

    See the config file for an example parameter sweep
    """
    # save the cfg to a global variable so it can be accessed anywhere
    args = utils.getArgs()
    setupEnvVariables(args)
    run = initWandb(args)

    # log some example loss
    for i in range(100):
        wandb.log({"loss": i})

    # log a message that training is complete
    print("Done!")
    print(wandb.config["learning_rate"])

    xSumDataset = xSum.XSum()
    summarizer = Summarizer()


if __name__ == "__main__":
    main()
