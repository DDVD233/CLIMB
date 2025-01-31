import wfdb
import numpy as np
from scipy.io import savemat
from scipy import signal
import os
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecg_conversion.log'),
        logging.StreamHandler()
    ]
)


def load_wfdb_record(record_path):
    """
    Load a WFDB record and return the signal data and metadata.
    """
    try:
        record = wfdb.rdrecord(record_path)
        return record.p_signal, record.__dict__
    except Exception as e:
        logging.error(f"Error loading record {record_path}: {str(e)}")
        return None, None


def preprocess_ecg(signals, sampling_rate=500):
    """
    Preprocess ECG signals with standard filtering and normalization.
    """
    if signals is None:
        return None

    try:
        # Remove baseline wander using high-pass filter
        nyquist = sampling_rate / 2
        cutoff = 0.5  # Hz
        b, a = signal.butter(3, cutoff / nyquist, 'high')
        signals_filtered = signal.filtfilt(b, a, signals, axis=0)

        # Remove high-frequency noise and powerline interference
        cutoff_low = 45  # Hz
        b, a = signal.butter(3, cutoff_low / nyquist, 'low')
        signals_filtered = signal.filtfilt(b, a, signals_filtered, axis=0)

        # Normalize each lead independently
        signals_normalized = np.zeros_like(signals_filtered)
        for i in range(signals_filtered.shape[1]):
            lead = signals_filtered[:, i]
            # Z-score normalization
            signals_normalized[:, i] = (lead - np.mean(lead)) / np.std(lead)

        return signals_normalized
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        return None


def process_single_record(record_path, output_base_dir):
    """
    Process a single ECG record and save as MAT file.
    """
    try:
        # Create output directory structure
        record_path = Path(record_path)
        relative_path = record_path.relative_to(record_path.parents[3])  # relative to 'files' directory
        output_path = Path(output_base_dir) / relative_path.parent
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{record_path.stem}.mat"

        # Skip if output file already exists
        if output_file.exists():
            logging.info(f"Skipping existing file: {output_file}")
            return True

        # Load and process the record
        signals, metadata = load_wfdb_record(str(record_path.with_suffix('')))
        if signals is None:
            return False

        signals_processed = preprocess_ecg(signals)
        if signals_processed is None:
            return False

        # Prepare data structure for MAT file
        data_dict = {
            'signals': signals_processed,
            'sampling_rate': metadata['fs'],
            'lead_names': metadata['sig_name'],
            'units': metadata['units'],
            'patient_id': metadata['record_name'],
            'date': metadata['base_date'],
            'time': metadata['base_time']
        }

        # Save to MAT file
        savemat(output_file, data_dict)
        return True

    except Exception as e:
        logging.error(f"Error processing {record_path}: {str(e)}")
        return False


def find_dat_files(base_dir):
    """
    Find all .dat files in the MIMIC-IV ECG directory structure.
    """
    base_path = Path(base_dir)
    return list(base_path.rglob('*.dat'))


def main():
    # Define directories
    input_base_dir = os.path.expanduser('~/high_modality/ecg/mimiciv')
    output_base_dir = os.path.expanduser('~/high_modality/ecg/mimiciv_processed')

    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Find all .dat files
    logging.info("Finding all ECG records...")
    dat_files = find_dat_files(os.path.join(input_base_dir, 'files'))
    total_files = len(dat_files)
    logging.info(f"Found {total_files} ECG records to process")

    # Process files using multiprocessing
    num_cores = mp.cpu_count() - 1  # Leave one core free
    logging.info(f"Processing using {num_cores} CPU cores")

    # Create processing pool
    with mp.Pool(num_cores) as pool:
        process_func = partial(process_single_record, output_base_dir=output_base_dir)

        # Process files with progress bar
        results = list(tqdm(
            pool.imap(process_func, dat_files),
            total=total_files,
            desc="Processing ECG records"
        ))

    # Summary
    successful = sum(1 for x in results if x)
    failed = total_files - successful
    logging.info(f"Processing complete!")
    logging.info(f"Successfully processed: {successful}")
    logging.info(f"Failed: {failed}")


if __name__ == "__main__":
    main()