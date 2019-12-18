# Python STL
import os
import sys
import argparse
# Data Science
import matplotlib.pyplot as plt
# PyTorch
import torch

# Local
from torchseg.model import model
from torchseg.trainer import Trainer


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
                        default=3,
                        help='Number of epochs')
    parser.add_argument('-v', '--val_freq', dest='val_freq', type=int,
                        default=5,
                        help='Validation frequency')

    parser_args = parser.parse_args()

    # Some checks on provided args
    if not os.path.isdir(os.path.join("torchseg", "checkpoints",
                                      parser_args.checkpoint_path)):
        raise FileNotFoundError("The checkpoints file {} was not found."
                                "Check the name again."
                                .format(parser_args.checkpoint_name))
    if parser_args.num_epochs <= 0:
        raise ValueError("Number of epochs must be > 0")

    if parser_args.num_workers < 0:
        raise ValueError("Number of workers must be >= 0")

    if parser_args.lr < 0:
        raise ValueError("Learning rate must be >= 0")

    if (parser_args.val_freq <= 0 or
            parser_args.val_freq > parser_args.num_epochs):
        raise ValueError("Validation frequency must be > 0 and "
                         "less than number of epochs")

    return parser_args


if __name__ == "__main__":
    # CLI
    args = cli()

    # Get trainer
    model_trainer = Trainer(model, args)
    # Start training + validation
    # Save model before exiting if there is a keyboard interrupt
    try:
        model_trainer.start()
    except KeyboardInterrupt as e:
        print("Exit requested during train-val")
        state = {
            "epoch": model_trainer.current_epoch,
            "best_loss": model_trainer.best_loss,
            "state_dict": model_trainer.net.state_dict(),
            "optimizer": model_trainer.optimizer.state_dict(),
        }
        print("******** Saving state before exiting ********")
        try:
            torch.save(state, os.path.join("torchseg",
                                           "checkpoints",
                                           args.checkpoint_name))
        except FileNotFoundError as e:
            print(f"Error while saving checkpoint\n{e}")
        sys.exit(0)

    # Get losses & scores from trainer
    losses = model_trainer.losses
    dice_scores = model_trainer.dice_scores
    iou_scores = model_trainer.iou_scores

    # Helper function to plot scores
    def plot(scores, name):
        plt.figure(figsize=(15, 5))
        plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
        plt.plot(range(len(scores["val"])), scores["val"], label=f'val {name}')
        plt.title(f'{name} plot')
        plt.xlabel('Epoch')
        plt.ylabel(f'{name}')
        plt.legend()
        plt.show()

    # Plot losses and scores
    plot(losses, "Loss")
    plot(dice_scores, "Dice score")
    plot(iou_scores, "IoU score")
