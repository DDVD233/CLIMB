import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DRIVE(Dataset):
    '''
    A dataset class for the DRIVE: Digital Retinal Images for Vessel Extraction Dataset.
    (https://drive.grand-challenge.org/)

    The DRIVE database consists of 40 retinal images from a diabetic retinopathy screening program:
    33 showing no signs of retinopathy and 7 showing mild early signs.

    The images, captured using a Canon CR5 camera, are divided into 20 training images with a single
    manual segementation of the vasculature available and 20 test images (no annotations available).

    Mask images indicating the field of view are provided for all images.
    '''

    # Dataset information
    INPUT_SIZE = (224, 224)

    def __init__(self, img_dir: str, mask_dir: str, manual_dir: str = None, train = True):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.manual_dir = manual_dir

        self.images = sorted([f for f in os.listdir(img_dir)])
        self.masks = sorted([f for f in os.listdir(mask_dir)])
        self.train = train

        # Manual annotations only available during training
        if train and manual_dir is not None:
            self.manuals = sorted([f for f in os.listdir(manual_dir)])
        else:
            self.manuals = None

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path)

        # Load mask
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = Image.open(mask_path)

        # Load manual segmentation if in training mode
        if self.train:
            manual_path = os.path.join(self.manual_dir, self.manuals[idx])
            manual = Image.open(manual_path)
        else:
            manual = None

        image = self.transform(image)
        mask = self.transform(mask)
        if manual is not None:
            manual = self.transform(manual)

        # Return based on training or testing mode
        if self.train:
            return image, mask, manual
        else:
            return image, mask
        
if __name__ == '__main__':

    # Training
    img_dir = './datasets/training/images'
    mask_dir = './datasets/training/mask'
    manual_dir = './datasets/training/1st_manual'

    drive_train = DRIVE(img_dir, mask_dir, manual_dir, True)
    image, mask, manual = drive_train[0]
    print(image.shape, mask.shape, manual.shape)

    dataloader_train = DataLoader(drive_train, batch_size=32, shuffle=True)




