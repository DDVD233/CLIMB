import os

from kaggle.api.kaggle_api_extended import KaggleApi


def any_exist(files):
    return any(map(os.path.exists, files))


class KaggleDownloader:
    """Handles authentication and downloading of files from Kaggle using KaggleApi"""

    def __init__(self, remote_path):
        self.remote_path = remote_path
        self.api = KaggleApi()
        self.api.authenticate()

    def download_file(self, local_path):
        """Download from kaggle"""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        download_path = local_path
        os.makedirs(download_path, exist_ok=True)
        self.api.dataset_download_files(self.remote_path, path=download_path, unzip=True)
