"""Utility functions for DICOM files"""

import cv2
import pydicom


def read_dicom(file_path: str, imsize: int):
    """Read pixel array from a DICOM file and apply recale and resize
    operations.

    The rescale operation is defined as the following:
        x = x * RescaleSlope + RescaleIntercept
    The rescale slope and intercept can be found in the DICOM files.

    Args:
        file_path (str): Path to a dicom file.
        resize_shape (int): Height and width for resizing.

    Returns:
        The processed pixel array from a DICOM file.
    """

    # read dicom
    dcm = pydicom.dcmread(file_path)
    pixel_array = dcm.pixel_array

    # rescale
    intercept = dcm.RescaleIntercept
    slope = dcm.RescaleSlope
    pixel_array = pixel_array * slope + intercept

    # resize
    resize_shape = imsize
    pixel_array = cv2.resize(pixel_array, (resize_shape, resize_shape), interpolation=cv2.INTER_AREA)

    return pixel_array
