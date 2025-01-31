import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DMR_Dataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=transforms.ToTensor()):
        self.csv_data = pd.read_csv(os.path.join(img_dir, csv_file))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir, self.csv_data.iloc[idx, 0])
        image = Image.open(img_path)
        davis_grade_concat = self.csv_data.iloc[idx, 1]
        davis_grade_single = self.csv_data.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)

        return image, davis_grade_concat, davis_grade_single


if __name__ == "__main__":
    dmr_dataset = DMR_Dataset(img_dir='dmr/', csv_file='list.csv')
    example_dmr_dataloader = DataLoader(dmr_dataset, batch_size=32, shuffle=True)
    
    for images, grades_concat, grades_single in example_dmr_dataloader:
        print(images.shape, grades_concat, grades_single)
