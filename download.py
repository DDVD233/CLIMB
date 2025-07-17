#!/usr/bin/env python3
"""
Global Dataset Downloader for CLIMB

Downloads all supported medical imaging datasets to a specified base directory.

Currently supported datasets:
- MIMIC CXR (MIMIC_CXR)
- VINDR CXR (VINDR_CXR)
- VINDR MAMMO (VINDR)
- BUSI (BUSI)
- COVID19 (Covid19Dataset)
- Brain Tumor (BrainTumorDataset)
- Brain Tumor 2 (BinaryBrainTumorDataset)
- CoronaHack (CoronaHackDataset)
- COVID BLUES (COVIDBLUES)
- COVID US (COVIDUS)
"""

import argparse
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import dataset classes
from src.datasets.chest_xray.mimic_cxr import MIMIC_CXR
from src.datasets.chest_xray.vindr_cxr import VINDR_CXR
from src.datasets.mammo.vindr import VINDR
from src.datasets.ultrasound.busi import BUSI
from src.datasets.chest_xray.covid19 import Covid19Dataset
from src.datasets.mri.brain_tumor import BrainTumorDataset
from src.datasets.mri.brain_tumor_2 import BinaryBrainTumorDataset
from src.datasets.chest_xray.coronahack import CoronaHackDataset
from src.datasets.ultrasound.covid_blue import COVIDBLUES
from src.datasets.ultrasound.covid_us import COVIDUS
from src.datasets.derm.HAM10000 import HAM10000
from src.datasets.derm.pad_ufes_20 import pad_ufes_20
from src.datasets.derm.isic2020 import ISIC2020
from src.datasets.ct.hemorrhage import BrainCTHemorrhageDataset
from src.datasets.ct.rspect import Rspect

# Dataset registry with display names and classes
DOWNLOADABLE_DATASETS = {
    'mimic_cxr': {
        'name': 'MIMIC CXR',
        'class': MIMIC_CXR,
    },
    'vindr_cxr': {
        'name': 'VINDR CXR',
        'class': VINDR_CXR,
    },
    'vindr_mammo': {
        'name': 'VINDR MAMMO',
        'class': VINDR,
    },
    'busi': {
        'name': 'BUSI',
        'class': BUSI,
    },
    'covid19': {
        'name': 'COVID19',
        'class': Covid19Dataset,
    },
    'brain_tumor': {
        'name': 'Brain Tumor',
        'class': BrainTumorDataset,
    },
    'brain_tumor_2': {
        'name': 'Brain Tumor 2',
        'class': BinaryBrainTumorDataset,
    },
    'coronahack': {
        'name': 'CoronaHack',
        'class': CoronaHackDataset,
    },
    'covid_blues': {
        'name': 'COVID BLUES',
        'class': COVIDBLUES,
    },
    'covid_us': {
        'name': 'COVID US',
        'class': COVIDUS,
    },
    'ham10000': {
        'name': 'HAM10000',
        'class': HAM10000,
    },
    'pad_ufes_20': {
        'name': 'PAD UFES 20',
        'class': pad_ufes_20,
    },
    'isic2020': {
        'name': 'ISIC 2020',
        'class': ISIC2020,
    },
    'brain_ct_hemorrhage': {
        'name': 'Brain CT Hemorrhage',
        'class': BrainCTHemorrhageDataset,
    },
    'rspect': {
        'name': 'RSPECT',
        'class': Rspect,
    },
}


def download_dataset(dataset_key, base_dir, args):
    """Download a single dataset."""
    dataset_info = DOWNLOADABLE_DATASETS[dataset_key]
    dataset_name = dataset_info['name']
    dataset_class = dataset_info['class']
    
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Create dataset instance with download=True
        # Pass base_dir directly - each dataset will handle its own subdirectory structure
        _ = dataset_class(base_root=base_dir, download=True)
            
        print(f"✓ Successfully downloaded {dataset_name}")
        
    except Exception as e:
        print(f"✗ Failed to download {dataset_name}")
        print(f"  Error: {str(e)}")
        if args.continue_on_error:
            print("  Continuing with next dataset...")
        else:
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Download all supported medical imaging datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported datasets:
{}
        """.format('\n'.join(f"  - {info['name']} ({key})" for key, info in DOWNLOADABLE_DATASETS.items()))
    )
    
    parser.add_argument(
        'base_dir',
        type=str,
        help='Base directory where all datasets will be downloaded'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=list(DOWNLOADABLE_DATASETS.keys()) + ['all'],
        default=['all'],
        help='Specific datasets to download (default: all)'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue downloading other datasets if one fails'
    )
    
    args = parser.parse_args()
    
    # Resolve base directory
    base_dir = os.path.abspath(args.base_dir)
    
    # Determine which datasets to download
    if 'all' in args.datasets:
        datasets_to_download = list(DOWNLOADABLE_DATASETS.keys())
    else:
        datasets_to_download = args.datasets
    
    print(f"Global Dataset Downloader")
    print(f"Base directory: {base_dir}")
    print(f"Datasets to download: {len(datasets_to_download)}")
    
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Download each dataset
    successful = 0
    failed = 0
    
    for dataset_key in datasets_to_download:
        try:
            download_dataset(dataset_key, base_dir, args)
            successful += 1
        except Exception as e:
            failed += 1
            if not args.continue_on_error:
                print(f"\nAborted due to error. Use --continue-on-error to skip failed downloads.")
                break
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Download Summary")
    print(f"{'='*60}")
    print(f"Total datasets: {len(datasets_to_download)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()