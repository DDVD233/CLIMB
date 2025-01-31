import json
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Ground truth are masks
class Kvasir_Dataset(Dataset):
    def __init__(self, img_dir='images', mask_dir='masks', bbox_json='kvasir_bboxes.json',
                 transform=transforms.ToTensor(), include_bbox_info=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        with open(bbox_json, 'r') as file:
            self.bbox_data = json.load(file)

        self.image_names = list(self.bbox_data)
        self.transform = transform
        self.include_bbox_info = include_bbox_info

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, image_name + '.jpg')
        image = Image.open(img_path)
        mask_path = os.path.join(self.mask_dir, image_name + '.jpg')
        mask = Image.open(mask_path)
        bbox_info = self.bbox_data[image_name]["bbox"]

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        if self.include_bbox_info:
            return image, mask, bbox_info

        return image, mask

if __name__ == "__main__":
    kvasir_dataset = Kvasir_Dataset(include_bbox_info=True)
    example_kvasir_dataloader = DataLoader(kvasir_dataset, batch_size=1, shuffle=True)
    
    # Note that images have different sizes
    
    for u in example_kvasir_dataloader:
        print(u[0].shape, u[1].shape, u[2])
        
