import wfdb
import numpy as np
from scipy.io import savemat
from scipy import signal
import os
from pathlib import Path
import datetime


def load_wfdb_record(record_path):
    """
    Load a WFDB record and return the signal data and metadata.

    Args:
        record_path (str): Path to the WFDB record without extension

    Returns:
        tuple: (signals, fields) where signals is numpy array and fields is metadata
    """
    record = wfdb.rdrecord(record_path)
    return record.p_signal, record.__dict__


def preprocess_ecg(signals, sampling_rate=500):
    """
    Preprocess ECG signals with standard filtering and normalization.

    Args:
        signals (numpy.ndarray): Raw ECG signals (n_samples x n_leads)
        sampling_rate (int): Sampling frequency in Hz

    Returns:
        numpy.ndarray: Preprocessed ECG signals
    """
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


def convert_to_mat(input_path, output_path):
    """
    Convert WFDB record to MAT file with preprocessing.

    Args:
        input_path (str): Path to input WFDB record without extension
        output_path (str): Path to output MAT file
    """
    # Load the record
    signals, metadata = load_wfdb_record(input_path)

    # Preprocess the signals
    signals_processed = preprocess_ecg(signals)

    # base_time: datetime.time(9, 24); base_date: datetime.date(2180, 1, 1)
    combined_date = datetime.datetime.combine(metadata['base_date'], metadata['base_time'])
    str_date = combined_date.strftime("%Y-%m-%d %H:%M:%S")
    # Prepare data structure for MAT file
    data_dict = {
        'signals': signals_processed,
        'sampling_rate': metadata['fs'],
        'lead_names': metadata['sig_name'],
        'units': metadata['units'],
        'patient_id': metadata['record_name'],
        'date': str_date
    }

    # Save to MAT file
    savemat(output_path, data_dict)


def batch_process_directory(input_dir, output_dir):
    """
    Process all WFDB records in a directory and its subdirectories.

    Args:
        input_dir (str): Input directory containing WFDB records
        output_dir (str): Output directory for MAT files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all .hea files (WFDB header files)
    for header_file in input_path.rglob('*.hea'):
        # Get the record path without extension
        record_path = str(header_file.parent / header_file.stem)

        # Create corresponding output path
        relative_path = header_file.relative_to(input_path)
        output_file = output_path / relative_path.parent / f"{header_file.stem}.mat"

        # Create output subdirectories if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            print(f"Processing: {record_path}")
            convert_to_mat(record_path, str(output_file))
            print(f"Successfully converted to: {output_file}")
        except Exception as e:
            print(f"Error processing {record_path}: {str(e)}")


def main():
    """
    Main function to demonstrate usage of the converter.
    """
    # Example usage
    input_dir = "/home/dvd/high_modality/ecg/mimiciv/files"
    output_dir = "/home/dvd/high_modality/ecg/mimiciv/mats"

    print("Starting batch conversion...")
    batch_process_directory(input_dir, output_dir)
    print("Conversion complete!")


if __name__ == "__main__":
    main()