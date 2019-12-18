# Python STL
import argparse
# Data Science
import matplotlib.pyplot as plt

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

    parser_args = parser.parse_args()

    return parser_args


if __name__ == "__main__":
    # CLI
    args = cli()

    # Get trainer
    model_trainer = Trainer(model, args)
    # Start training + validation
    model_trainer.start()

    # Plot training scores and losses
    losses = model_trainer.losses
    dice_scores = model_trainer.dice_scores
    iou_scores = model_trainer.iou_scores

    def plot(scores, name):
        plt.figure(figsize=(15, 5))
        plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
        plt.plot(range(len(scores["val"])), scores["val"], label=f'val {name}')
        plt.title(f'{name} plot')
        plt.xlabel('Epoch')
        plt.ylabel(f'{name}')
        plt.legend()
        plt.show()

    plot(losses, "Loss")
    plot(dice_scores, "Dice score")
    plot(iou_scores, "IoU score")
