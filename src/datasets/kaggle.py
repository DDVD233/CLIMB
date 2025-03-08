import getpass
import json
import os
import subprocess
import gzip
import shutil

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.utils import extract_archive
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from urllib.parse import urljoin

from src.datasets.specs import Input2dSpec



from kaggle.api.kaggle_api_extended import KaggleApi


def any_exist(files):
    return any(map(os.path.exists, files))



class KaggleDownloader:
    """Handles authentication and downloading of files from Kaggle using KaggleApi"""

    def __init__(self, remote_path): #, base_url):
        #self.base_url = base_url
        self.remote_path = remote_path
        self.credentials_file = os.path.expanduser("~/.kaggle/kaggle.json")
        self._load_or_prompt_credentials()

    def _load_or_prompt_credentials(self):
        """Load credentials from file or prompt user"""
        if os.path.exists(self.credentials_file):
            with open(self.credentials_file, 'r') as f:
                self.username = f.readline().strip()
                self.password = f.readline().strip()
        else:
            print("Kaggle credentials not found. Please enter them now:")
            self.username = input("Username: ")
            self.password = getpass.getpass("Key: ")
            # Save credentials
            os.makedirs(os.path.dirname(self.credentials_file), exist_ok=True)
            with open(self.credentials_file, 'w') as f:
                f.write(f'{{"username":"{username}","key":"{password}"}}')
            os.chmod(self.credentials_file, 0o600)  # Secure the credentials file

    def download_file(self, local_path):
        """Download from kaggle"""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Construct the full URL
        #url = urljoin(self.base_url, remote_path)

        api = KaggleApi()
        api.authenticate()

        download_path = local_path
        os.makedirs(download_path, exist_ok=True)
        api.dataset_download_files(self.remote_path, path=download_path, unzip=True)








        
