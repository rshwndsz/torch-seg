# Python STL
import os
import argparse
import logging
# Data Science
import matplotlib.pyplot as plt
# Image processing
import cv2
# PyTorch
import torch
from torch.utils.data import Dataset
# Data Augmentations
from albumentations.augmentations import transforms as T
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

# Local
from torchseg.data import DATA_FOLDER
from torchseg.model import model

_DIRNAME = os.path.dirname(__file__)


class TestDataset(Dataset):
    """API for the test dataset

    Attributes
    ----------
    root : str
        Root folder of the dataset
    image_names : list[str]
        Sorted list of test images
    transform : albumentations.core.composition.Compose
        Albumentations augmentations pipeline
    """
    def __init__(self, data_folder):
        self.root = data_folder
        self.image_names = sorted(os.listdir(os.path.join(self.root, "test", "imgs")))
        self.transform = Compose(
            [
                T.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), p=1),
                T.Resize(256, 256),
                ToTensorV2(),
            ]
        )

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.root, "test", "imgs", image_name)
        image = cv2.imread(image_path)
        images = self.transform(image=image)["image"]
        return images

    def __len__(self):
        return len(self.image_names)


def cli():
    parser = argparse.ArgumentParser(description='Torchseg')
    parser.add_argument('-c', '--checkpoint', dest='checkpoint_name', type=str,
                        default="model.pth",
                        help='Name of checkpoint file inside torchseg/checkpoints/')

    parser_args = parser.parse_args()

    # Validate provided args
    test_checkpoint_path = os.path.join("torchseg", "checkpoints",
                                        parser_args.checkpoint_name)
    if not os.path.exists(test_checkpoint_path):
        raise FileNotFoundError("The checkpoints file at {} was not found."
                                "Check the name again."
                                .format(checkpoint_path))
    else:
        logger.info(f"Loading checkpoint file: {checkpoint_path}")

    return parser_args


# TODO: Write code for overlapping window evaluation (replace downsampling)
if __name__ == "__main__":
    # Get logger
    logger = logging.getLogger(__name__)

    # Parse CLI arguments
    args = cli()

    # Get test dataset
    testset = TestDataset(DATA_FOLDER)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Switch model over to evaluation model
    model.eval()
    # Load state from checkpoint
    checkpoint_path = os.path.join(_DIRNAME, "torchseg",
                                   "checkpoints", args.checkpoint_name)
    state = torch.load(checkpoint_path)
    model.load_state_dict(state["state_dict"])

    # For every image in testset, predict and plot
    with torch.no_grad():
        for i, batch in enumerate(testset):
            batch = batch.unsqueeze(dim=0)
            probs = torch.sigmoid(model(batch.to(device)))
            preds = (probs > 0.5).float()
            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(preds.cpu().numpy().squeeze(), 'gray')
            ax[1].imshow(batch.cpu().numpy().squeeze().transpose(1, 2, 0), 'gray')
            ax[0].set_title("Prediction")
            ax[1].set_title("Image")
            plt.show()
