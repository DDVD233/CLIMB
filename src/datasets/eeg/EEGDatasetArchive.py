import os

import numpy as np
import pandas as pd
from mne.filter import resample, notch_filter
from numpy.ma.core import arange
import pickle as pkl
import json
from mne_bids import (BIDSPath, read_raw_bids)
import mne
from scipy.io import loadmat
import matplotlib.pyplot as plt

from src.datasets.eeg.utils import count_dirs_starting_with, count_bids_runs, \
    visualize_epochs_channels, bandpass_filter_raw, identify_bad_channels_and_interpolate, resample_raw, \
    notch_filter_raw, reference_eeg_raw


class PhysioDataset:

    x_dict_name = 'x_dict.pkl'
    y_dict_name = 'y_dict.pkl'
    metadata_name = 'metadata.pkl'

    def __init__(self, raw_data_root_dir, processed_data_root_dir, dataset_name, load_from_exist = False, save_dataset = True, **kwargs):

        self.raw_data_root_dir = raw_data_root_dir
        assert os.path.exists(self.raw_data_root_dir), f"Path {self.raw_data_root_dir} does not exist"
        self.processed_data_root_dir = processed_data_root_dir
        assert os.path.exists(self.processed_data_root_dir), f"Path {self.processed_data_root_dir} does not exist"
        self.dataset_name = dataset_name

        self.raw_data_dir = os.path.join(self.raw_data_root_dir, self.dataset_name)
        self.save_data_dir = os.path.join(self.processed_data_root_dir, self.dataset_name)

        self.load_from_exist = load_from_exist
        self.save_dataset = save_dataset

        # place holders for x and y
        self.x_dict = {}
        self.y_dict = {}
        self.metadata = {}

        # # initialize x and y from the x_dict and y_dict
        # self.x, self.y = self.initialize_x_y()


    def load_data_dict(self):
        if self.load_from_exist:
            return self.load_physio_dict()
        else:
            x_dict, y_dict, metadata = self.get_dataset_x_y_metadata_dict()

            if self.save_dataset:
                # save the data dict into the disk
                # create the save_data_dir
                if not os.path.exists(self.save_data_dir):
                    os.makedirs(self.save_data_dir)
                x_dict_path = os.path.join(self.save_data_dir, self.x_dict_name)
                pkl.dump(x_dict, open(x_dict_path, 'wb'))

                y_dict_path = os.path.join(self.save_data_dir, self.y_dict_name)
                pkl.dump(y_dict, open(y_dict_path, 'wb'))

                metadata_path = os.path.join(self.save_data_dir, self.metadata_name)
                pkl.dump(metadata, open(metadata_path, 'wb'))

            return x_dict, y_dict, metadata


    def load_physio_dict(self):
        # load the pickle file from the processed_data_root_dir
        x_dict_path = os.path.join(self.processed_data_root_dir, self.dataset_name, self.x_dict_name)
        assert os.path.exists(x_dict_path), f"Path {x_dict_path} does not exist"
        y_dict_path = os.path.join(self.processed_data_root_dir, self.dataset_name, self.y_dict_name)
        assert os.path.exists(y_dict_path), f"Path {y_dict_path} does not exist"
        metadata_path = os.path.join(self.processed_data_root_dir, self.dataset_name, self.metadata_name)
        assert os.path.exists(metadata_path), f"Path {metadata_path} does not exist"

        x_dict = pkl.load(open(x_dict_path, 'rb'))
        y_dict = pkl.load(open(y_dict_path, 'rb'))
        metadata = pkl.load(open(metadata_path, 'rb'))

        return x_dict, y_dict, metadata

    def get_dataset_x_y_metadata_dict(self):
        x_dict = {}
        y_dict = {}
        metadata = {}
        # need to implement the function to get the dataset from the root dir

        return x_dict, y_dict, metadata

    def get_participants_ids(self):
        return self.x_dict.keys()

    def get_participant_x_dict(self, participant_id):
        return self.x_dict[participant_id]

    def get_participant_y_dict(self, participant_id):
        return self.y_dict[participant_id]

    def initialize_x_y(self, **kwargs):
        return np.array([]), np.array([])


class FACED(PhysioDataset):

    def __init__(self, raw_data_root_dir, processed_data_root_dir, dataset_name, load_from_exist = False, save_dataset = True, **kwargs):
        super().__init__(raw_data_root_dir, processed_data_root_dir, dataset_name, load_from_exist, save_dataset)

        self.metadata = kwargs
        self.x_dict, self.y_dict, self.metadata = self.load_data_dict()

    def get_dataset_x_y_metadata_dict(self):

        x_dict = {}
        y_dict = {}

        metadata = self.metadata
        foldPaths = os.path.join(self.raw_data_dir, 'Recording_info.csv')
        subs = pd.read_csv(foldPaths, low_memory=False)['sub']

        for sub in subs:

            sub_path = os.path.join(self.raw_data_dir, 'Processed_data',sub + '.pkl')
            assert os.path.exists(sub_path), f"Path {sub_path} does not exist"
            data = pd.read_pickle(sub_path)

            sub_x = data
            sub_y = arange(len(data))

            # have another layer for dictionary for eeg

            x_dict[sub] = {'eeg':{ 'run-1': sub_x}}
            y_dict[sub] = {'eeg':{ 'run-1': sub_y}}

        return x_dict, y_dict, metadata


    def initialize_x_y(self):
        # TODO: implement the function to initialize x and y from the x_dict and y_dict
        # need to split the data into smaller windows, this will depend on how the model is implemented and the classification task
        return np.array([]), np.array([])


class AuditoryOddballDelorme2020(PhysioDataset):

    def __init__(self, raw_data_root_dir, processed_data_root_dir, dataset_name, load_from_exist = False, save_dataset = True, **kwargs):
        super().__init__(raw_data_root_dir, processed_data_root_dir, dataset_name, load_from_exist, save_dataset)

        self.metadata = kwargs
        self.x_dict, self.y_dict, self.metadata = self.load_data_dict()

    def get_dataset_x_y_metadata_dict(self):
        # need to implement the function to get the dataset from the root dir
        x_dict = {}
        y_dict = {}
        metadata = self.metadata

        print('get_dataset_x_y_metadata_dict')

        datatype = self.metadata['bids_config']['datatype']
        task = self.metadata['bids_config']['task']
        extension = self.metadata['bids_config']['extension']
        suffix = self.metadata['bids_config']['suffix']
        bids_root = self.raw_data_dir


        montage_name = metadata['preprocessing_config']['montage_name']
        low_cut = metadata['preprocessing_config']['filter_config']['low_cut']
        high_cut = metadata['preprocessing_config']['filter_config']['high_cut']
        notch_freqs = metadata['preprocessing_config']['filter_config']['notch_freqs']
        resample_fs = metadata['preprocessing_config']['resample_fs']
        epoch_t_min = metadata['preprocessing_config']['epoch_window']['t_min']
        epoch_t_max = metadata['preprocessing_config']['epoch_window']['t_max']
        include_t_max = metadata['preprocessing_config']['epoch_window']['include_t_max']
        if not include_t_max:
            epoch_t_max = epoch_t_max - 1 / resample_fs

        baseline = metadata['preprocessing_config']['baseline']

        montage = mne.channels.make_standard_montage(montage_name)

        label = metadata['label']

        num_subjects, sub_dirs = count_dirs_starting_with(self.raw_data_dir, 'sub-')

        for sub_num, sub_dir in enumerate(sub_dirs):

            subject = sub_dir.split("-")[1]
            sub_path = os.path.join(bids_root, sub_dir)
            bids_path = BIDSPath(root=bids_root, datatype=datatype)
            # get the number of runs for this subject
            data_path = os.path.join(sub_path, datatype)
            num_runs, run_ids = count_bids_runs(data_path)

            sub_x_dict = {'eeg': {}}
            sub_y_dict = {'eeg': {}}
            for run_id in run_ids:

                bids_path = bids_path.update(subject=subject, task=task, run=run_id, suffix=suffix, extension=extension)
                raw = read_raw_bids(bids_path=bids_path, verbose=False)
                # set the montage, this is not necessary for this dataset, because the channel names are in the raw data
                raw.set_montage(montage)
                # preprocess the raw data

                # resample the raw data
                raw = resample_raw(raw, resample_fs)

                # filter data
                raw = bandpass_filter_raw(raw, low_cut, high_cut, picks='eeg')
                raw = notch_filter_raw(raw, freqs=notch_freqs, picks='eeg')

                # discover and interpolate the bad channels
                raw = identify_bad_channels_and_interpolate(raw, thresh1=3, proportion=0.3, picks='eeg')

                # re-reference the raw data
                raw = reference_eeg_raw(raw)



                # get the event channels and event dict
                events, event_id = mne.events_from_annotations(raw)

                # we need remap the event id back to the master event id
                try:
                    remapped_event_id = {key: label[key] for key in event_id.keys()}
                except KeyError as e:
                    print(f"KeyError: {e}")
                    print(f"event_id: {event_id}")
                    print(f"master_event_id: {label}")
                    raise
                # change the local event id to the master event id which is same as the 'label' in the metadata
                for i in range(len(events)):
                    # If the original event in the matrix matches the old event_id,
                    # replace it with the new event_id from the master_event_id dictionary
                    event_find = False
                    for events_key, events_value in event_id.items():
                        if events[i, 2] == events_value:
                            events[i, 2] = label[events_key]
                            event_find = True
                            break
                    if not event_find:  # the remapped event_id is not found
                        print(f"Event not found in event_id: {events[i, 2]}")
                        # we need exit the program with exception
                        raise ValueError("Event not found in event_id")

                epochs = mne.Epochs(raw, events, remapped_event_id, tmin=epoch_t_min, tmax=epoch_t_max,
                                    baseline=baseline, preload=True)


                # uncomment this to visualize the epochs

                visualize_epochs_channels(epochs,
                                          event_groups={"stimulus/standard": 2, "stimulus/oddball_with_reponse": 8},
                                          colors={"stimulus/standard": "blue", "stimulus/oddball_with_reponse": "red"},
                                          picks=['Fz', 'Cz', 'Pz', 'Oz'],
                                          tmin_vis=-0.1, tmax_vis=0.8,
                                          title=f'Auditory Oddball Delorme 2020 - Participant',
                                          out_dir=None, verbose='INFO', fig_size=(12.8, 7.2))


                run_data = epochs.get_data(picks='eeg')
                run_labels = epochs.events[:, -1]

                sub_x_dict['eeg']['run-' + run_id] = run_data
                sub_y_dict['eeg']['run-' + run_id] = run_labels

            x_dict['sub' + subject] = sub_x_dict
            y_dict['sub' + subject] = sub_y_dict

        metadata['ch_names'] = montage.ch_names

        return x_dict, y_dict, metadata

    def initialize_x_y(self):
        # TODO: construct the x and y from the x_dict, and y_dict. Usually we only interested in correct target and correct non-target
        pass

class BCIIV2a(PhysioDataset):
    """

    This class is used to get the BCICIVA dataset
    Dataset location: https://www.bbci.de/competition/iv/
    Description: https://www.bbci.de/competition/iv/desc_2a.pdf

    """

    def __init__(self, raw_data_root_dir, processed_data_root_dir, dataset_name, load_from_exist=False,
                 save_dataset=True, **kwargs):
        super().__init__(raw_data_root_dir, processed_data_root_dir, dataset_name, load_from_exist, save_dataset)

        self.metadata = kwargs
        self.x_dict, self.y_dict, self.metadata = self.load_data_dict()


    # note: there are two runs, the first run will be used as the training data, and the second run will be used as the testing data
    def get_dataset_x_y_metadata_dict(self):

        x_dict = {}
        y_dict = {}
        metadata = self.metadata

        ch_names = metadata['ch_names']
        ch_names = ch_names + ["EOG1", "EOG2", "EOG3"]
        ch_types = ['eeg'] * 22 + ['eog'] * 3
        montage_name = metadata['preprocessing_config']['montage_name']
        resample_fs = metadata['preprocessing_config']['resample_fs']
        epoch_t_min = metadata['preprocessing_config']['epoch_window']['t_min']
        epoch_t_max = metadata['preprocessing_config']['epoch_window']['t_max']
        include_t_max = metadata['preprocessing_config']['epoch_window']['include_t_max']

        if not include_t_max:
            epoch_t_max = epoch_t_max - 1 / resample_fs

        montage = mne.channels.make_standard_montage(montage_name)
        # Note: the data has been filtered already with bandpass filter and notch filter. Refer to the paper for more details.

        print('get_dataset_x_y_metadata_dict')
        subject_num = 9
        subject_list = ['A0' + str(i) for i in range(1, subject_num + 1)]
        sessions = ['T', 'E']

        for subject in subject_list:

            sub_x_dict = {'T': {}, 'E': {}}
            sub_y_dict = {'T': {}, 'E': {}}


            for session in sessions:
                session_dict_x = {}
                session_dict_y = {}

                # read the mat file
                data_path = os.path.join(self.raw_data_dir, f'{subject}{session}.mat')
                assert os.path.exists(data_path), f"Path {data_path} does not exist"
                sessions_data = loadmat(data_path)


                # there are 6 runs. the experiment starting from the fourth run
                for run in range(3, 9):
                    # note: the A04T.mat only has two EOG runs, so we need to skip it
                    run_index = run
                    if subject == 'A04' and session == 'T':
                        run_index = run - 2

                    run_data = sessions_data['data'][0][run_index]
                    X = run_data['X'][0][0]
                    trail = run_data['trial'][0][0]
                    y = run_data['y'][0][0]
                    fs = run_data['fs'][0][0]
                    artifacts = run_data['artifacts'][0][0]

                    # construct the mne raw object.

                    info = mne.create_info(ch_names, ch_types = ch_types, sfreq=fs)
                    info.set_montage(montage)

                    raw = mne.io.RawArray(X.T, info)

                    # create the events
                    events = np.zeros((len(trail), 3))
                    events[:, 0] = trail[:, 0]
                    events[:, 2] = y[:, 0]

                    # remove the row where the artifact is 1
                    events = events[artifacts[:, 0] == 0]
                    events = events.astype(int)
                    # create the epochs
                    epochs = mne.Epochs(raw, events, tmin=epoch_t_min, tmax=epoch_t_max, baseline=(epoch_t_min, epoch_t_min+(epoch_t_max-epoch_t_min)*0.1), preload=True)

                    data = epochs.get_data(picks='eeg')
                    labels = epochs.events[:, -1]


                    session_dict_x['run-' + str(run)] = data
                    session_dict_y['run-' + str(run)] = labels

                sub_x_dict[session]['eeg'] = session_dict_x
                sub_y_dict[session]['eeg'] = session_dict_y

            x_dict[subject] = sub_x_dict
            y_dict[subject] = sub_y_dict

        return x_dict, y_dict, metadata







if __name__ == '__main__':
    # try get FACED dataset

    raw_data_root_dir = 'C:/Dataset/raw'
    processed_data_root_dir = 'C:/Dataset/processed'

    # read the json file
    with open('eeg_dataset_config.json') as f:
        config = json.load(f)

    dataset_name = 'FACED'
    dataset_metadata = config['FACED']
    faced_dataset = FACED(raw_data_root_dir, processed_data_root_dir, dataset_name, load_from_exist=False, save_dataset=True, **dataset_metadata)
    #
    dataset_name = 'auditory_oddball_delorme2020'
    dataset_metadata = config['auditory_oddball_delorme2020']
    auditory_oddball_delorme2020 = AuditoryOddballDelorme2020(raw_data_root_dir, processed_data_root_dir, dataset_name, load_from_exist=False, save_dataset=True, **dataset_metadata)
    #
    dataset_name = 'BCIIV2a'
    dataset_metadata = config['BCIIV2a']
    bciciva_dataset = BCIIV2a(raw_data_root_dir, processed_data_root_dir, dataset_name, load_from_exist=False, save_dataset=True, **dataset_metadata)


    print("Finished")