import json
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MedPix_Dataset(Dataset):
    def __init__(self, img_dir, descriptions_json, transform=transforms.ToTensor()):
        self.img_dir = img_dir
        with open(descriptions_json, 'r') as file:
            self.descriptions = json.load(file)

        self.desc_list = list(self.descriptions)

        self.transform = transform

    def __len__(self):
        return len(self.desc_list)

    def __getitem__(self, idx):
        description = self.desc_list[idx]
        img_path = os.path.join(self.img_dir, description['image'] + '.png')
        image = Image.open(img_path)
        desc = description['Description']

        acr_codes = desc['ACR Codes']
        age = desc['Age']
        caption = desc['Caption']
        figure_part = desc['Figure Part']

        if figure_part == None:
            figure_part = ''
        
        modality = desc['Modality']
        plane = desc['Plane']
        sex = desc['Sex']

        if self.transform:
            image = self.transform(image)

        return image, acr_codes, age, caption, figure_part, modality, plane, sex

if __name__ == "__main__":
    medpix_dataset = MedPix_Dataset(img_dir='images/', descriptions_json='Descriptions.json')
    example_medpix_dataloader = DataLoader(medpix_dataset, batch_size=1, shuffle=True)
    
    # Note that images have different sizes
    
    for u in example_medpix_dataloader:
        print(u[0].shape, u[3])
