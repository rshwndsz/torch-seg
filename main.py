# Python STL
import os
import sys
import argparse
import logging
from logging.config import dictConfig
# Data Science
import matplotlib.pyplot as plt
# PyTorch
import torch

# Local
from torchseg.model import model
from torchseg.trainer import Trainer

_DIRNAME = os.path.dirname(__file__)
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%H:%M:%S'
        },
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        'torchseg': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        '__main__': {  # if __name__ == '__main__'
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
    }
}

# Load logging configuration from dict LOGGING_CONFIG
dictConfig(LOGGING_CONFIG)
# Create logger
logger = logging.getLogger(__name__)


def cli():
    parser = argparse.ArgumentParser(description='Torchseg')
    parser.add_argument('-c', '--checkpoint', dest='checkpoint_name', type=str,
                        default="model.pth",
                        help='Name of checkpoint file inside torchseg/checkpoints/')
    parser.add_argument('-b', '--batch_size', dest="batch_size", type=int,
                        default=32,
                        help='Batch size')
    parser.add_argument('-w', '--workers', dest='num_workers', type=int,
                        default=4,
                        help='Number of workers for dataloader')
    parser.add_argument('--lr', dest='lr', type=float,
                        default=3e-4,
                        help='Initial learning rate')
    parser.add_argument('-e', '--epoch', dest='num_epochs', type=int,
                        default=10,
                        help='Number of epochs')
    parser.add_argument('-v', '--val_freq', dest='val_freq', type=int,
                        default=5,
                        help='Validation frequency')

    parser_args = parser.parse_args()

    # Some checks on provided args
    checkpoint_path = os.path.join(_DIRNAME, "torchseg", "checkpoints",
                                   parser_args.checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("The checkpoints file at {} was not found."
                                "Check the name again."
                                .format(checkpoint_path))
    else:
        logger.info(f"Loading checkpoint file: {checkpoint_path}")

    if parser_args.num_epochs <= 0:
        raise ValueError("Number of epochs: {} must be > 0"
                         .format(parser_args.num_epochs))
    else:
        logger.info(f"Number of epochs: {parser_args.num_epochs}")

    if parser_args.num_workers < 0:
        raise ValueError("Number of workers: {} must be >= 0"
                         .format(parser_args.num_workers))
    else:
        logger.info(f"Number of workers: {parser_args.num_workers}")

    if parser_args.lr < 0:
        raise ValueError("Learning rate: must be >= 0"
                         .format(parser_args.lr))
    else:
        logger.info(f"Initial Learning rate: {parser_args.lr}")

    if (parser_args.val_freq <= 0 or
            parser_args.val_freq > parser_args.num_epochs):
        raise ValueError("Validation frequency: {} must be > 0 and "
                         "less than number of epochs"
                         .format(parser_args.val_freq))
    else:
        logger.info(f"Validation frequency: {parser_args.val_freq} epochs")

    return parser_args


if __name__ == "__main__":
    # CLI
    args = cli()

    # Get trainer
    model_trainer = Trainer(model, args)
    # Save model before exiting if there is a keyboard interrupt
    try:
        # Start training + validation
        model_trainer.start()
    except KeyboardInterrupt or SystemExit as e:
        logger.info("Exit requested during train-val")
        # Collect state
        state = {
            "epoch": model_trainer.current_epoch,
            "best_loss": model_trainer.best_loss,
            "state_dict": model_trainer.net.state_dict(),
            "optimizer": model_trainer.optimizer.state_dict(),
        }
        logger.info("******** Saving state before exiting ********")
        # Save state if possible
        try:
            torch.save(state, os.path.join("torchseg",
                                           "checkpoints",
                                           args.checkpoint_name))
        except FileNotFoundError as e:
            logger.error(f"Error while saving checkpoint\n{e}")
        sys.exit(0)

    # Helper function to plot scores
    def metric_plot(scores, name):
        plt.figure(figsize=(15, 5))
        plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
        plt.plot(range(len(scores["val"])), scores["val"], label=f'val {name}')
        plt.title(f'{name} plot')
        plt.xlabel('Epoch')
        plt.ylabel(f'{name}')
        plt.legend()
        plt.show()

    for metric_name, metric_values in model_trainer.holder.store.items():
        metric_plot(metric_values, metric_name)
