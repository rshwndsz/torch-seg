# Python STL
import os
import argparse
# Data Science
import numpy as np
import matplotlib.pyplot as plt
# Image processing
import cv2
# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader
# Data Augmentations
from albumentations.augmentations import transforms as T
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

# Local
from torchseg.data import DATA_FOLDER
from torchseg.model import model

_DIRNAME = os.path.dirname(__file__)


class TestDataset(Dataset):
    def __init__(self, data_folder, tta=4):
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

    return parser_args


# TODO: Write code for overlapping window evaluation (replace downsampling)
if __name__ == "__main__":
    args = cli()
    testset = TestDataset(DATA_FOLDER)

    device = torch.device("cuda")

    model.eval()
    checkpoint_path = os.path.join(_DIRNAME, "torchseg", "checkpoints", args.checkpoint_name)
    state = torch.load(checkpoint_path)
    model.load_state_dict(state["state_dict"])

    with torch.no_grad():
        for i, batch in enumerate(testset):
            batch = batch.unsqueeze(dim=0)
            print(batch.shape)
            probs = torch.sigmoid(model(batch.to(device)))
            preds = (probs > 0.5).float()
            plt.imshow(preds.cpu().numpy().squeeze(), 'gray')
            plt.show()
            plt.imshow(batch.cpu().numpy().squeeze().transpose(1, 2, 0), 'gray')
            plt.show()
