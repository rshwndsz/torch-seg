# Python STL
import os
# Image Processing
import cv2
# PyTorch
import torch
from torch.utils.data import DataLoader, Dataset, sampler
# Data augmentation
from albumentations.augmentations import transforms as T
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

# Root folder of dataset
DATA_FOLDER = "../input/dataset/raw/"


# TODO: Try to create a BasicDataset class you can inherit from
class OrganDataset(Dataset):
    def __init__(self, data_folder, phase):
        """
        Create an API for the dataset
        :param data_folder: Path to root folder of the dataset
        :param phase: Phase of learning; In ['train', 'val']
        """
        # Root folder of the dataset
        assert os.path.isdir(data_folder), "{} is not a directory or it doesn't exist.".format(data_folder)
        self.root = data_folder

        # Phase of learning
        assert phase in ['train', 'test', 'val'], "Provide any one of train/test/val as phase."
        self.phase = phase

        # Data Augmentations and tensor transformations
        self.transforms = get_transforms(self.phase)

        # Get names & number of images in root/train or root/val
        _path_to_imgs = os.path.join(self.root, self.phase, "imgs")
        assert os.path.isdir(_path_to_imgs), "{} doesn't exist.".format(_path_to_imgs)
        self.image_names = sorted(os.listdir(_path_to_imgs))
        assert len(self.image_names) != 0, "No images found in {}".format(_path_to_imgs)

    def __getitem__(self, idx):
        # Load image
        image_name = self.image_names[idx]
        image_path = os.path.join(self.root, self.phase, "imgs", image_name)
        image = cv2.imread(image_path)
        assert image.size != 0, "cv2: Unable to load image - {}".format(image_path)

        # Load mask
        mask_name = image_name
        mask_path = os.path.join(self.root, self.phase, "masks", mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # <<<< Change: Generalize reading rgb/gray
        assert mask.size != 0, "cv2: Unable to load mask - {}".format(mask_path)

        # TODO: Improve this spagetti (ノಠ益ಠ)ノ彡┻━┻
        # Data Augmentation for image and mask
        augmented = self.transforms['aug'](image=image, mask=mask)
        new_image = self.transforms['img_only'](image=augmented['image'])
        new_mask = self.transforms['mask_only'](image=augmented['mask'])
        aug_tensors = self.transforms['final'](image=new_image['image'], mask=new_mask['image'])
        image = aug_tensors['image']
        mask = aug_tensors['mask']
        mask = torch.unsqueeze(mask, dim=0)  # For [1, H, W] instead of [H, W]
        return image, mask

    def __len__(self):
        return len(self.image_names)


# TODO: Make it easier to add augmentations
# TODO: Add logging here
# TODO: Move into DataSet as static method
def get_transforms(phase):
    """
    Get composed albumentations transforms
    :param phase: Phase of learning; In ['train', 'val']
    :return: Composed list of albumentations transforms
    """
    aug_transforms = []

    if phase == "train":
        # Data augmentation for training only
        aug_transforms.extend([
            T.ShiftScaleRotate(
                shift_limit=0,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5),
            T.Flip(p=0.5),
            T.RandomRotate90(p=0.5),
        ])
        # Exotic Augmentations for train only 🤤
        aug_transforms.extend([
            T.RandomBrightnessContrast(p=0.5),
            T.ElasticTransform(p=0.5),
            T.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=True, p=0.2),
            T.JpegCompression(quality_lower=95, quality_upper=100, p=0.4),  # <<< Note: Deprecated
            T.Blur(blur_limit=7, p=0.3),
        ])
    aug_transforms.extend([
        T.RandomSizedCrop(min_max_height=(256, 256),
                          height=256,
                          width=256,
                          w2h_ratio=1.0,
                          interpolation=cv2.INTER_LINEAR,
                          p=1.0),
    ])
    aug_transforms = Compose(aug_transforms)

    mask_only_transforms = Compose([
        T.Normalize(mean=0, std=1, always_apply=True)
    ])
    image_only_transforms = Compose([
        T.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), always_apply=True)
    ])
    final_transforms = Compose([
        ToTensorV2()
    ])

    transforms = {
        'aug': aug_transforms,
        'img_only': image_only_transforms,
        'mask_only': mask_only_transforms,
        'final': final_transforms
    }
    return transforms


# TODO: Add logging here
def provider(data_folder, phase, batch_size=8, num_workers=4):
    """
    Return DataLoader for the Dataset
    :param data_folder: Path to root folder of the dataset
    :param phase: Phase of learning; In ['train', 'val']
    :param batch_size: Batch size; Usually a multiple of 8
    :param num_workers: Number of workers; Saturate your shared mem
    :return: DataLoader for the provided phase
    """
    image_dataset = OrganDataset(data_folder, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    return dataloader
