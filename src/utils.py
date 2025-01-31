import os

import numpy as np
import torch
from pydicom import dcmread

LABEL_FRACS = {'small': 8, 'medium': 64, 'large': 256, 'full': np.inf}



def jinchi_align(preds):
    # print(preds.shape)
    preds[:,1] = preds[:,1] + preds[:,2]
    return torch.cat([preds[:, :2], preds[:, 3:]], dim=1)

def align_jinchi_labels(preds, labels, num_jinchi_labels):
    '''
    Account for label differences between Jinchi University dataset and other ophthalmology datasets.
    Messidor2 (training dataset) and Aptos use the standard Davis scale with 5 classes. 
    Two of those classes are combined into one for the modified Davis scale used by Jinchi.
    This method thus takes predictions based on the standard Davis scale, and combines classes 1 and 2 into
    class 1 for the modified Davis scale.
    '''
    # create new preds tensor for modified Davis scale
    jinchi_preds = torch.zeros((preds.shape[0], num_jinchi_labels), device=preds.device)
    for i in range(preds.shape[0]):
        # if modified Davis class is 1, and predicted label is standard Davis class 1 or 2, convert prediction to class 1
        if labels[i] == 1 and (torch.argmax(preds[i]) == 1 or torch.argmax(preds[i]) == 2):
            jinchi_preds[i][1] = 1
        # otherwise, shift predicted label down by one (except class 0)
        else:
            jinchi_preds[i][max(0, torch.argmax(preds[i])-1)] = 1
    return jinchi_preds


def get_pixel_array(dcm_file):
    '''
    Convert DICOM pixel array into image format.
    Adapted from https://stackoverflow.com/questions/42650233/how-to-access-rgb-pixel-arrays-from-dicom-files-using-pydicom
    '''
    dcm = dcmread(dcm_file)
    img = dcm.pixel_array

    # Rescale pixel array
    if hasattr(dcm, 'RescaleSlope'):
        img = img * dcm.RescaleSlope
    if hasattr(dcm, 'RescaleIntercept'):
        img = img + dcm.RescaleIntercept

    # Convert pixel_array (img) to -> gray image (img_2d_scaled)
    ## Step 1. Convert to float to avoid overflow or underflow losses.
    img_2d = img.astype(float)

    ## Step 2. Rescaling grey scale between 0-255
    img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0

    ## Step 3. Convert to uint
    img_2d_scaled = np.uint8(img_2d_scaled)

    ## Step 4. Invert pixels if MONOCHROME1
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        img_2d_scaled = np.invert(img_2d_scaled)

    return img_2d_scaled


def count_files(dir_path):
    count = 0
    for _, _, files in os.walk(dir_path):
        count += len(files)
    return count