import cv2
import nibabel as nib
import numpy as np


def apply_window_level(data, window=700, level=80):
    """
    Apply radiological window/level adjustment optimized for CTPA
    window: controls the range of values displayed (contrast)
    level: controls the center of the window (brightness)
    Typical CTPA window/level values are around 700/80 to visualize both
    lung parenchyma and contrast-enhanced vessels while reducing noise
    """
    lower = level - window / 2
    upper = level + window / 2
    data_adj = np.clip(data, lower, upper)
    data_adj = ((data_adj - lower) / (window)) * 255
    return data_adj.astype(np.uint8)


def denoise_and_enhance(image):
    """
    Apply denoising and subtle enhancement
    """
    # Convert to uint8 if not already
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

    # Apply bilateral filter for edge-preserving denoising
    # denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    try:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        return enhanced
    except:
        return image


def nifti_to_video(nifti_path, output_path, axis=0, fps=10):
    """
    Convert a NIfTI file to video along specified axis with optimized CTPA enhancement
    """
    # Load the NIfTI file
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata()

    # Get dimensions for the video
    if axis == 0:
        frames = data.shape[0]
        height, width = data.shape[1:3]
    elif axis == 1:
        frames = data.shape[1]
        height, width = data.shape[0], data.shape[2]
    else:  # axis == 2
        frames = data.shape[2]
        height, width = data.shape[0:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each slice
    for i in range(frames):
        if axis == 0:
            frame = data[i, :, :]
        elif axis == 1:
            frame = data[:, i, :]
        else:  # axis == 2
            frame = data[:, :, i]

        # Apply windowing optimized for CTPA
        frame = apply_window_level(frame, window=700, level=80)
        frame = denoise_and_enhance(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Write the frame
        out.write(frame_rgb)

    # Release the video writer
    out.release()
