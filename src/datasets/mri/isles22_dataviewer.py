import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ISLES22_Dataset(Dataset):
    '''
    data_path is the path to the raw data (should be same as root folder)
    ground_truth_path is the path to the ground truth (should be derivatives folder)
    There are 3 different types of raw data images: FLAIR, adc, and dwi
    Ground truth images are msk
    
    Directory structure:
    derivatives
        sub-strokecase0001
            ses-0001
                msk image (ground truth)
        ... (one folder for each participant)
        sub-strokecase0250
            ...
    
    sub-strokecase0001
        ses-0001
            anat
                FLAIR image
            dwi
                adc image
                dwi image
    ... (one folder for each participant)
    sub-strokecase0250
        ...
    
    participants_file is a file that contains basic information for each participant
        such as the participant id (e.g. sub-strokecase0001), sex, age, and weight
    
    transform is the transformation function, which by default turns the data into
        tensor format
    '''
    def __init__(self, data_path='./', ground_truth_path='derivatives',
                 participants_file='participants.tsv',
                 image_folders=['anat', 'dwi', 'dwi'],
                 image_types=['FLAIR', 'adc', 'dwi'],
                 gt_name='msk',
                 transform=transforms.ToTensor()):
        self.data_path = data_path
        self.ground_truth_path = ground_truth_path
        self.image_folders = image_folders
        self.image_types = image_types
        self.gt_name = gt_name
        self.transform = transform

        participants_info = pd.read_csv(participants_file, sep='\t')
        self.participant_ids = participants_info['participant_id']

    def __len__(self):
        return len(self.participant_ids)

    def __getitem__(self, idx):
        s = 'ses-0001'
        participant_id = self.participant_ids[idx]
        d_path = os.path.join(self.data_path, participant_id + '/', s + '/')
        g_path = os.path.join(self.ground_truth_path, participant_id + '/',
                         s + '/')
        data_list = []
        for t_idx in range(len(self.image_types)):
            d_file = os.path.join(d_path, self.image_folders[t_idx] + '/',
                                  participant_id + '_' + s + '_'
                                      + self.image_types[t_idx] + '.nii.gz')
            data_list.append(self.transform(nib.load(d_file).get_fdata()))

        g_file = os.path.join(g_path, participant_id + '_' + s + '_'
                                  + self.gt_name + '.nii.gz')
        data_list.append(self.transform(nib.load(g_file).get_fdata()))

        return tuple(data_list)


if __name__ == "__main__":
    isle_dataset = ISLES22_Dataset()
    example_isle_dataloader = DataLoader(isle_dataset, batch_size=1, shuffle=True)

    # data points have different sizes
    for flair_img, adc_img, dwi_img, gt_img in example_isle_dataloader:
        print(flair_img.shape, adc_img.shape, dwi_img.shape, gt_img.shape)





