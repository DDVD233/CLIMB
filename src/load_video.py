import cv2
import numpy as np


def load_video(video_path) -> np.ndarray:
    """
    Load a video from a file path. The video is loaded as a list of frames in numpy.ndarray format in RGB.
    Args:
        video_path (str): Path to the video file.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    frames = np.stack(frames)
    return frames
