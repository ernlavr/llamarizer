import logging
import os
import wandb
import argparse
import transformers
import src.utils.utilities as utils

from omegaconf import DictConfig, OmegaConf
from huggingface_hub.hf_api import HfFolder


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def setupEnvVariables():
    os.environ['WANDB_API_KEY'] = '3e554d2dafac7c22b4e348afff4b69a9e6d49e81'
    os.environ['WANDB_PROJECT'] = 'adv_nlp2023'
    os.environ["WANDB_LOG_MODEL"] = 'checkpoint' # 'checkpoint' or 
    os.environ['HF_TOKEN'] = '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'
    HfFolder.save_token(os.environ['HF_TOKEN'])

def initWandb():
    wandb.init(project="adv_nlp2023")

def main():
    """ Main entry point for the software. Essentially able to deligate execution to
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
    setupEnvVariables()
    args = utils.getArgs()
    initWandb()
    print(args)

    # log some example loss
    for i in range(100):
        wandb.log({"loss": i})

    # log a message that training is complete
    wandb.log({"message": "Training complete!"})
    print("Done!")
    print(args.no_wandb)

    # set log level to debug
    log.setLevel(logging.DEBUG)
    log.error("This is an error message")
    log.info("This is an info message")



if __name__ == "__main__":
    main()