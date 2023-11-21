import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    # model hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')

    # other stuff
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    return parser.parse_args()