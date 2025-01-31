"""Base CT dataset class"""

from typing import Union

import h5py
import numpy as np
from torch.utils.data import Dataset


class CTDatasetBase(Dataset):
    """A base dataset class for CT scan datasets"""

    def __init__(self, train: bool):

        self.hdf5_dataset = None

        if train:
            self.split = "train"
        else:
            self.split = "eval"

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def num_classes():
        raise NotImplementedError

    @staticmethod
    def spec():
        raise NotImplementedError

    def windowing(self, pixel_array: np.array, window_center: int, window_width: int):
        """
        Adjust pixel array of CT scans to a particular viewing window. The
        windowing opertaion will change the appearance of the image to highlight
        particular anatomical structures.

        The upperbound of the viewing window is calcuated as:
            WindowCetner + WindowWidth // 2
        While the lowerbound of the viewing window is:
            WindowCetner - WindowWidth // 2

        More information can be found here:
            https://radiopaedia.org/articles/windowing-ct

        Args:
            pixel_array (np.array): Pixel array of a CT scan.
            window_center (int): Midpoint of the range of the CT numbers displayed.
            window_width (int): The range of CT numbers that an image contains
        """

        lower = window_center - window_width // 2
        upper = window_center + window_width // 2
        pixel_array = np.clip(pixel_array.copy(), lower, upper)
        pixel_array = (pixel_array - lower) / (upper - lower)

        return pixel_array

    def is_float(self, element: any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    def read_from_hdf5(self, hdf5_path: str, key: str, slice_idx: Union[int, list] = None):
        """read pre-processed CT from HDF5 file

        Args:
            hdf5_path (str): path to hdf5 file
            key (str): study name
            slice_idx (int | list): index(es) from study to load
        """

        # load HDF5 file to memory for parallelization
        if self.hdf5_dataset is None:
            self.hdf5_dataset = h5py.File(hdf5_path, "r")
        if slice_idx is None:
            arr = self.hdf5_dataset[str(key)][:]
        else:
            arr = self.hdf5_dataset[str(key)][slice_idx]
        return arr
