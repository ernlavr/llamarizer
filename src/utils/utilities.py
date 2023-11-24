import argparse
import omegaconf

def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--args_path', type=str, default=None, help='Path to args file')
    
    # model hyperparameters
    parser.add_argument('--learning_rate',  type=float, default=1e-5,   help='Learning rate')
    parser.add_argument('--optimizer',      type=str,   default='adam', help='Optimizer')
    parser.add_argument('--weight_decay',   type=float, default=0.01,   help='Weight decay')
    parser.add_argument('--epochs',         type=int,   default=1,      help='Number of epochs')
    parser.add_argument('--batch_size',     type=int,   default=32,      help='Batch size')
    parser.add_argument('--sequence_length', type=int,  default=512,    help='Sequence length')
    parser.add_argument('--model_name',     type=str,   default='gpt2', help='Model name')
    parser.add_argument('--wandb_mode',     type=str,   default='online', help='Wandb mode')

    # other stuff
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')

    # check if we have a config file
    args = parser.parse_args()
    if args.args_path is not None:
        # load the config file
        args = omegaconf.OmegaConf.load(args.args_path)
        # convert to argparse
        args = omegaconf.OmegaConf.to_container(args, resolve=True)

    return args