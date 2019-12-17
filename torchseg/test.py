# Python STL
import os
# Image processing
import cv2
# PyTorch
from torch.utils.data import Dataset, DataLoader
# Data Augmentations
from albumentations.augmentations import transforms as T
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

# Local
from .data import DATA_FOLDER


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


test_loader = DataLoader(TestDataset(DATA_FOLDER),
                         batch_size=4,
                         shuffle=False,
                         num_workers=2,
                         pin_memory=True,
                         )

# TODO: Write code for evaluation with weight-loading

# model = model_trainer.net # get the model from model_trainer object
# model.eval()
# state = torch.load('checkpoints/model-2.pth', map_location=lambda storage, loc: storage)
# model.load_state_dict(state["state_dict"])
# encoded_pixels = []
# for i, batch in enumerate(tqdm(testset)):
#     preds = torch.sigmoid(model(batch.to(device)))
#     preds = preds.detach().cpu().numpy().squeeze()
#     preds = np.where(preds>0.5, 1, preds)
#     preds = np.where(preds<=0.5, 0, preds)
#     print(preds.shape)
#     plt.imshow(preds, 'gray')
#     plt.show()
#     plt.imshow(batch.cpu().numpy().squeeze().transpose(1, 2, 0))
#     plt.show()
