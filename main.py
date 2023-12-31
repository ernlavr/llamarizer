import logging
import os
import wandb
import src.utils.utilities as utils
from src.ml.summarizer import Summarizer
import src.datasets.xSum as xSum
import src.ml.nli as nli
import src.ml.summarizerEvaluation as se

from huggingface_hub.hf_api import HfFolder


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def setupEnvVariables(args):
    os.environ["WANDB_API_KEY"] = "ENTER_KEY_HERE"
    os.environ["WANDB_PROJECT"] = "adv_nlp2023"
    os.environ["WANDB_MODE"] = args.wandb_mode
    #os.environ["WANDB_LOG_MODEL"] = "end"  # 'checkpoint' or
    os.environ["HF_TOKEN"] = "ENTER_TOKEN_HERE"
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
    print("Loading args")
    args = utils.getArgs()
    print("Setup env variables")
    setupEnvVariables(args)
    print("Init wandb")
    run = initWandb(args)
    
    if args.train_summ:
        print("Train summarizer")
        summarizer = Summarizer()
        summarizer.train()
    
    if args.train_nli:
        print("Train NLI")
        model = nli.NLI_Finetune()
        model.finetune()

    if args.eval_summ:
        print("Eval summarizer")
        evaluator = se.LlamarizerEval()
        evaluator.eval()
    


if __name__ == "__main__":
    main()
