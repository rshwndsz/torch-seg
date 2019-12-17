# Data Science
import matplotlib.pyplot as plt

# Local
from torchseg.model import model
from torchseg.trainer import Trainer


if __name__ == "__main__":
    # TODO: Add CLI with argparse
    # Get trainer
    model_trainer = Trainer(model)
    # Start training + validation
    model_trainer.start()

    # Plot training scores and losses
    losses = model_trainer.losses
    dice_scores = model_trainer.dice_scores
    iou_scores = model_trainer.iou_scores

    def plot(scores, name):
        plt.figure(figsize=(15, 5))
        plt.plot(range(len(scores["train"][12:])), scores["train"][12:], label=f'train {name}')
        plt.plot(range(len(scores["train"][12:])), scores["val"][12:], label=f'val {name}')
        plt.title(f'{name} plot')
        plt.xlabel('Epoch')
        plt.ylabel(f'{name}')
        plt.legend()
        plt.show()

    plot(losses, "Loss")
    plot(dice_scores, "Dice score")
    plot(iou_scores, "IoU score")
