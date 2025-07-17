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

    def download_file(self, local_path, type='dataset'):
        """Download from kaggle"""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        download_path = local_path
        os.makedirs(download_path, exist_ok=True)
        try:
            if type == 'dataset':
                self.api.dataset_download_files(self.remote_path, path=download_path, unzip=True)
            elif type == 'competition':
                self.api.competition_download_files(self.remote_path, path=download_path)
        except:
            webpage = ""
            if type == 'dataset':
                webpage = "https://www.kaggle.com/datasets/" + self.remote_path
            elif type == 'competition':
                webpage = "https://www.kaggle.com/c/" + self.remote_path
            print(f"Failed to download {self.remote_path} to {download_path}. Please check the remote path and your Kaggle API credentials."
                  f"\nVisit {webpage} and apply for access if needed.")
            raise ValueError()
