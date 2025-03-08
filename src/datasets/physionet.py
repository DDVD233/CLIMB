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


def any_exist(files):
    return any(map(os.path.exists, files))



class PhysioNetDownloader:
    """Handles authentication and downloading of files from PhysioNet using wget"""

    def __init__(self, base_url="https://physionet.org/files/mimic-cxr-jpg/2.0.0/"):
        self.base_url = base_url
        self.credentials_file = os.path.expanduser("~/.physionet_credentials")
        self._load_or_prompt_credentials()

    def _load_or_prompt_credentials(self):
        """Load credentials from file or prompt user"""
        if os.path.exists(self.credentials_file):
            with open(self.credentials_file, 'r') as f:
                self.username = f.readline().strip()
                self.password = f.readline().strip()
        else:
            print("PhysioNet credentials not found. Please enter them now:")
            self.username = input("Username: ")
            self.password = getpass.getpass("Password: ")
            # Save credentials
            os.makedirs(os.path.dirname(self.credentials_file), exist_ok=True)
            with open(self.credentials_file, 'w') as f:
                f.write(f"{self.username}\n{self.password}")
            os.chmod(self.credentials_file, 0o600)  # Secure the credentials file

    def download_file(self, remote_path, local_path): #, is_folder=False):
        """Download a single file from PhysioNet using wget"""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Construct the full URL
        url = urljoin(self.base_url, remote_path)

        # Prepare wget command
        wget_command = [
            'wget',
            '-N',  # Only download if newer
            '-c',  # Continue partially downloaded files
            '--no-check-certificate',  # Skip certificate validation
            '--user', self.username,
            '--password', self.password,
            '-O', local_path,  # Output to specific file
            url
        ]

        #if is_folder:
        #    wget_command = [
        #        'wget',
        #        '-r',  # Recursive download
        #        '-N',  # Only download if newer
        #        '-c',  # Continue partially downloaded files
        #        '--no-check-certificate',  # Skip certificate validation
        #        '--user', self.username,
        #        '--password', self.password,
        #        '-P', local_path,  # Output to specific file
        #        url
        #    ]

        try:
            # Run wget command
            result = subprocess.run(
                wget_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            if result.returncode == 0:
                return True
            else:
                print(f"Error downloading {url}: {result.stderr}")
                return False

        except subprocess.SubprocessError as e:
            print(f"Error running wget: {str(e)}")
            return False
