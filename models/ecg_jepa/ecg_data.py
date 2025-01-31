import numpy as np
from utils import return_purified, return_purified_feature, return_unique
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import os
import wfdb
from tqdm import tqdm
import glob
import h5py
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

# Function to downsample waves using resampling
def downsample_waves(waves, new_size):
    return np.array([resample(wave, new_size, axis=1) for wave in waves])

def remove_invalid_samples(waves, index=False):
    """
    Remove samples with NaN values or samples with the first 15 timesteps being all zeros.
    
    Args:
    waves (numpy.ndarray): The input array with shape (n_samples, n_channels, n_timesteps).
    index (bool): If True, output is indices of valid samples; otherwise, output is the valid samples themselves.
    """
    # Remove samples with NaN values
    nan_mask = np.isnan(waves).any(axis=(1, 2))
    
    # Remove samples with all zeros in the first 15 timesteps
    zero_mask = (np.abs(waves[:, :, :15]).sum(axis=(1, 2)) == 0)
    
    # Combine masks to find valid samples
    valid_indices = ~(nan_mask | zero_mask)
    
    # Print the number of invalid samples
    n_invalid = np.sum(~valid_indices)
    print(f'invalid samples: {n_invalid}')
    
    if index:
        return valid_indices
    else:
        return waves[valid_indices]
    

# Custom dataset class
class ECGDataset(Dataset):
    def __init__(self, waves, labels, transform=None):
        self.waves = torch.tensor(waves, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        wave = self.waves[idx]
        label = self.labels[idx]

        if self.transform:
            wave = self.transform(wave)
        return wave, label

# Custom dataset class
class ECGDataset_pretrain(Dataset):
    def __init__(self, waves):
        self.waves = torch.tensor(waves, dtype=torch.float32)

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        wave = self.waves[idx]
        return wave

def extract_diagnosis_code(record):
    for comment in record.comments:
        if comment.startswith('Dx:'):
            return comment.split(': ')[1]
    return None

def get_ecg_data(data_dir, reduced_lead=True, use_more=False, dx=False):
    """
    Read ECG data from the specified directory and return the data as a numpy array.

    Args:
    data_dir (str): The directory containing the ECG data files.
    reduced_lead (bool): If True, only eight leads are used: I, II, V1, V2, V3, V4, V5, V6.
    use_more (bool): If True, ECGs with more than 10s of data are split into multiple segments of 10s each.
    dx (bool): If True, extract the diagnosis code from the comments in the header file.

    Returns:
    np.ndarray: The ECG data array with shape (n_samples, n_channels, n_timesteps).
    np.ndarray (optional): Diagnostic code if dx is True.
    """
    ecg_records = []
    ecg_labels = []
    segment_length = 5000  # Length of each segment in samples (10 seconds at 500 Hz)

    for filename in os.listdir(data_dir):
        if filename.endswith('.hea'):
            record_name = os.path.splitext(filename)[0]
            record = wfdb.rdrecord(os.path.join(data_dir, record_name))
            ecg_data = record.p_signal
            ecg_label = extract_diagnosis_code(record) if dx else None

            # Resample the data if the sampling frequency is not 500 Hz
            if record.fs != 500:
                # Calculate the new length for resampling
                new_length = int((500 / record.fs) * record.sig_len)
                ecg_data = resample(ecg_data, new_length)

            # Process the data in segments
            if ecg_data.shape[0] >= segment_length:
                ecg_records.append(ecg_data[:segment_length])
                ecg_labels.append(ecg_label)
    
    # Convert lists to numpy arrays
    ecg_records = np.stack(ecg_records, axis=0)
    ecg_records = ecg_records.transpose(0, 2, 1)  # (n_samples, n_channels, n_timesteps)
    ecg_labels = np.array(ecg_labels)

    if reduced_lead:
        # Keep only the leads I, II, V1, V2, V3, V4, V5, V6
        ecg_records = np.concatenate((ecg_records[:, :2, :], ecg_records[:, 6:, :]), axis=1)

    if dx:
        return ecg_records, ecg_labels
    else:
        return ecg_records


def subdirectory(data_dir):
    contents = os.listdir(data_dir)
    data_dirs = [d for d in contents if os.path.isdir(os.path.join(data_dir, d))]
    return data_dirs

def waves_cinc(data_dir, reduced_lead=True):
    waves = []
    for subdir in subdirectory(data_dir):
        for minibatch in subdirectory(os.path.join(data_dir, subdir)):
            ecg_data = get_ecg_data(os.path.join(data_dir, subdir, minibatch), reduced_lead=reduced_lead)
            waves.append(ecg_data)

    waves = np.concatenate(waves, axis=0)
    waves = remove_invalid_samples(waves)
    return waves

def waves_shao(data_dir, reduced_lead=True):
    waves = []
    for subdir in subdirectory(data_dir):
        for minibatch in subdirectory(os.path.join(data_dir, subdir)):
            ecg_data = get_ecg_data(os.path.join(data_dir, subdir, minibatch), reduced_lead=reduced_lead)
            waves.append(ecg_data)

    waves = np.concatenate(waves, axis=0)
    waves = remove_invalid_samples(waves)
    return waves

# def waves_shao(data_dir, reduced_lead=True):
#     waves = get_ecg_data(data_dir, reduced_lead=reduced_lead, dx=False)
#     waves = remove_invalid_samples(waves)
#     return waves

class Code15Dataset(Dataset):
    def __init__(self, data_dir, transform=None, reduced_lead=True, downsample=True, use_cache=True):
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(data_dir, '*.hdf5'))
        self.transform = transform
        self.reduced_lead = reduced_lead 
        self.downsample = downsample
        self.file_indices = []
        self._cache = {}

        # Cache file path
        self.cache_file = os.path.join(data_dir, 'file_indices_cache.npy')
         
        # Precompute the indices for each file and filter out padded waves
        self._compute_file_indices(use_cache)

    def _compute_file_indices(self, use_cache):
        if use_cache and os.path.exists(self.cache_file):
            self.file_indices = np.load(self.cache_file, allow_pickle=True).tolist()
        else:
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(self._process_file, enumerate(self.files)))
            for file_idx, indices in results:
                self.file_indices.extend([(file_idx, i) for i in indices])

            # Save the generated file indices to cache
            if use_cache:
                np.save(self.cache_file, np.array(self.file_indices, dtype=object))

    def _process_file(self, file_idx_and_name):
        file_idx, filename = file_idx_and_name
        valid_indices = []
        with h5py.File(filename, 'r') as f:
            num_samples = f['tracings'].shape[0]
            for i in range(num_samples):
                wave = np.array(f['tracings'][i])
                if not np.all(wave[:10] == 0):  # Check if first 10 timesteps are not all zeros
                    valid_indices.append(i)
        return file_idx, valid_indices

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        if idx >= len(self.file_indices):
            raise IndexError(f"Index {idx} out of range for dataset with length {len(self.file_indices)}")
        file_idx, sample_idx = self.file_indices[idx]
        filename = self.files[file_idx]

        # Check cache first
        if (file_idx, sample_idx) in self._cache:
            wave = self._cache[(file_idx, sample_idx)]
        else:
            with h5py.File(filename, 'r') as f:
                wave = np.array(f['tracings'][sample_idx])
            self._cache[(file_idx, sample_idx)] = wave  # Cache the loaded wave

        # Transpose the wave so channels come first
        wave = wave.T
        
        if self.reduced_lead:
            wave = wave[[0, 1, 6, 7, 8, 9, 10, 11], :]
        
        if self.downsample:
            wave = resample(wave, 2500, axis=1)

        if self.transform:
            wave = self.transform(wave)
            
        return torch.tensor(wave, dtype=torch.float)

def waves_ptbxl(data_dir, task='multilabel', reduced_lead=True, downsample=True):
    from ptbxl_utils import load_dataset, compute_label_aggregations, select_data
    assert task in ['multilabel', 'multiclass']

    cat = 'superdiagnostic'
    categories = ['all', 'diagnostic', 'subdiagnostic', 'superdiagnostic', 'form', 'rhythm']
    assert cat in categories, f'Invalid category: {cat}, choose from {categories}'

    sampling_frequency=500

    # Load PTB-XL data
    data, raw_labels = load_dataset(data_dir, sampling_frequency)
    data = data.transpose(0,2,1)
    
    if downsample:
        data = np.array([resample(data[i], 2500, axis=1) for i in range(len(data))])
    
    if reduced_lead:
        data = np.concatenate([data[:,:2], data[:,6:]], axis=1)

    # Preprocess label data
    labels = compute_label_aggregations(raw_labels, data_dir, cat)
    # Select relevant data and convert to one-hot
    data_, labels, Y, _ = select_data(data, labels, cat, min_samples=0)

    # 1-9 for training 
    waves_train = data_[labels.strat_fold < 10]
    labels_train = Y[labels.strat_fold < 10]

    # 10 for validation
    waves_test = data_[labels.strat_fold == 10]
    labels_test = Y[labels.strat_fold == 10]

    if task == 'multiclass':
        waves_train, labels_train = convert_to_multiclass(waves_train, labels_train)
        waves_test, labels_test = convert_to_multiclass(waves_test, labels_test)

    return waves_train, waves_test, labels_train, labels_test


def waves_cpsc(data_dir, task='multilabel', reduced_lead=True, downsample=True):
    waves_cpsc = []
    labels_cpsc = []
    minibatches = []

    for minibatch in subdirectory(data_dir):
        ecg_data, ecg_labels = get_ecg_data(os.path.join(data_dir, minibatch), reduced_lead=True,  dx=True)
        waves_cpsc.append(ecg_data)
        labels_cpsc.append(ecg_labels)
        minibatches.extend([minibatch] * len(ecg_data))

    waves_cpsc = np.concatenate(waves_cpsc, axis=0)
    labels_cpsc = np.concatenate(labels_cpsc, axis=0)
    minibatches = np.array(minibatches)

    # Remove samples with NaN values
    valid_indices = remove_invalid_samples(waves_cpsc, index=True)
    
    # remove samples with empty labels
    for i in range(len(labels_cpsc)):
        if labels_cpsc[i] == '':
            valid_indices[i] = False

    waves_cpsc = waves_cpsc[valid_indices]
    labels_cpsc = labels_cpsc[valid_indices]
    minibatches = minibatches[valid_indices]

    if downsample:
        waves_cpsc = downsample_waves(waves_cpsc, 2500)

    # Extract unique labels
    unique_labels = set()
    for label_str in np.unique(labels_cpsc):
        labels = label_str.split(',')
        unique_labels.update(labels)

    unique_labels = sorted(unique_labels)
    # Create a mapping from label to index
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    # Initialize the binary matrix
    labels_matrix = np.zeros((len(labels_cpsc), len(unique_labels)), dtype=int)

    # Populate the binary matrix
    for i, label_str in enumerate(labels_cpsc):
        labels = label_str.split(',')
        for label in labels:
            labels_matrix[i, label_to_index[label]] = 1

    labels_cpsc = labels_matrix

    test_indices = (minibatches == 'g7')
    train_indices = ~test_indices

    waves_train = waves_cpsc[train_indices]
    labels_train = labels_cpsc[train_indices]
    waves_test = waves_cpsc[test_indices]
    labels_test = labels_cpsc[test_indices]

    if task == 'multiclass':
        waves_train, labels_train = convert_to_multiclass(waves_train, labels_train)
        waves_test, labels_test = convert_to_multiclass(waves_test, labels_test)

    return waves_train, waves_test, labels_train, labels_test

def convert_to_multiclass(waves, labels):
    '''
    convert multi-label to multi-class by restricting to samples with only one label
    '''

    label_sums = np.sum(labels, axis=1)
    indices_with_one_label = np.where(label_sums == 1)[0]

    waves = waves[indices_with_one_label]
    labels = labels[indices_with_one_label]

    # ont-hot to integer
    labels = np.argmax(labels, axis=1)

    return waves, labels

def waves_from_config(config, reduced_lead=True):
    # model_name = config['model_name']
    data_dir = config['data_dir']
    dataset = config['dataset']
    task = config['task']

    # if model_name == 'st_mem':
    #     reduced_lead = False

    if dataset == 'ptbxl':
        waves_train, waves_test, labels_train, labels_test = waves_ptbxl(data_dir, task, reduced_lead=reduced_lead)

    elif dataset == 'cpsc':
        waves_train, waves_test, labels_train, labels_test = waves_cpsc(data_dir, task, reduced_lead=reduced_lead)

    # # st_mem needs shorter waves 
    # if model_name == 'st_mem':
    #     waves_train = waves_train[:, :, 125:-125]
    #     waves_test = waves_test[:, :, 125:-125]

    return waves_train, waves_test, labels_train, labels_test