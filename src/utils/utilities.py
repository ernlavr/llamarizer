import argparse
import omegaconf
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import transformers
import wandb
import numpy as np


def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--args_path", type=str, default=None, help="Path to args file")

    # Control
    parser.add_argument("--train_summ", type=bool, default=False, help="Train Summarizer")
    parser.add_argument("--train_nli", type=bool, default=False, help="Fine-tune NLI")
    parser.add_argument("--train_with_nli", type=bool, default=False, help="Train summarizer with NLI")
    parser.add_argument("--eval_summ", type=bool, default=False, help="Train Summarizer")

    # model hyperparameters
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--sequence_length", type=int, default=512, help="Sequence length"
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Model name")
    parser.add_argument("--wandb_mode", type=str, default="online", help="Wandb mode")
    parser.add_argument(
        "--load_in_4bit", action="store_true", help="Load in 4bit weights", default=False
    )
    parser.add_argument("--repetition_penalty", type=float, default=1.25, help="Repetition penalty")
    parser.add_argument("--eval_steps", type=int, default=2, help="Eval steps")

    # dataset hyperparameters
    parser.add_argument("--train_size", type=int, default=100, help="Number of datapoints in training set")
    parser.add_argument("--val_size", type=int, default=10, help="Number of datapoints in validation set")
    parser.add_argument("--upsample_train", type=bool, help="If true, upsample. False downsamples")
    parser.add_argument("--upsample_val", type=bool,help="If true, upsample. False downsamples")



    # other stuff
    parser.add_argument("--wandb_num_examples", type=int, default=4, help="Number of examples to log")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--use_prompt", type=bool, default=False, help="Wraps input text with a prompt")
    parser.add_argument("--additional_info", type=str, default=None, help="Prompt to use")
    parser.add_argument("--save_model_at_end", type=bool, default=False, help="Save model at end of training")

    # check if we have a config file
    args = parser.parse_args()
    if args.args_path is not None:
        # load the config file
        args = omegaconf.OmegaConf.load(args.args_path)
        # convert to argparse
        args = omegaconf.OmegaConf.to_container(args, resolve=True)
        # convert to namespace
        args = argparse.Namespace(**args)

    return args

    
def get_peft_config():
    return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none"
        )

def get_bnb_config():
    return transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )

def load_from_wandb(id, load_in_4bit, peft_config, bnb_config):
    # load the model from wandb
    artifact = wandb.run.use_artifact(id, type="model")
    artifact_dir = artifact.download()

    return get_model(artifact_dir, load_in_4bit, peft_config, bnb_config)


def get_model(id, load_in_4bit=False, peft_config=None, bnb_config=None):
    model = None
    if load_in_4bit:
        # TODO: review model loading according to llama-recipes
        model = transformers.AutoModelForCausalLM.from_pretrained(
            id, 
            quantization_config=bnb_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(id)

    # load tokenizer and return
    tokenizer = transformers.AutoTokenizer.from_pretrained(id)
    return model, tokenizer