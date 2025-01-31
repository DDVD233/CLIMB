import json
import os
import pickle as pkl

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.filter import notch_filter, resample
from mne_bids import BIDSPath, read_raw_bids
from numpy.ma.core import arange
from scipy.io import loadmat

from src.datasets.eeg.utils import (
    bandpass_filter_raw, count_bids_runs, count_dirs_starting_with, identify_bad_channels_and_interpolate, notch_filter_raw,
    reference_eeg_raw, resample_raw, visualize_epochs_channels
)


class PhysioDataset:

    data_dict_name = 'data_dict.pkl'
    metadata_name = 'metadata.pkl'

    def __init__(self, dataset_name, load_from_exist: bool, raw_data_root_dir=None, processed_data_root_dir=None, save_dataset = True, **kwargs):



        self.dataset_name = dataset_name
        self.load_from_exist = load_from_exist


        self.raw_data_root_dir = raw_data_root_dir
        self.processed_data_root_dir = processed_data_root_dir

        self.raw_data_dir = os.path.join(self.raw_data_root_dir, self.dataset_name)
        self.save_data_dir = os.path.join(self.processed_data_root_dir, self.dataset_name)

        self.save_dataset = save_dataset

        if load_from_exist:
            assert os.path.exists(self.processed_data_root_dir), f"Path {self.processed_data_root_dir} does not exist"

        if save_dataset:
            assert os.path.exists(self.raw_data_root_dir), f"Path {self.raw_data_root_dir} does not exist"



        # place holders for x and y
        self.data_dict = {}
        self.metadata = {}


    def load_data_dict(self):
        if self.load_from_exist:
            return self.load_physio_dict()
        else:
            data_dict, metadata = self.get_dataset_data_dict()

            if self.save_dataset:
                # save the data dict into the disk
                # create the save_data_dir
                if not os.path.exists(self.save_data_dir):
                    os.makedirs(self.save_data_dir)

                data_dict_path = os.path.join(self.save_data_dir, self.data_dict_name)
                pkl.dump(data_dict, open(data_dict_path, 'wb'))

                metadata_path = os.path.join(self.save_data_dir, self.metadata_name)
                pkl.dump(metadata, open(metadata_path, 'wb'))

            return data_dict, metadata


    def load_physio_dict(self):
        # load the pickle file from the processed_data_root_dir
        data_dict_path = os.path.join(self.processed_data_root_dir, self.dataset_name, self.data_dict_name)
        assert os.path.exists(data_dict_path), f"Path {data_dict_path} does not exist"

        metadata_path = os.path.join(self.processed_data_root_dir, self.dataset_name, self.metadata_name)
        assert os.path.exists(metadata_path), f"Path {metadata_path} does not exist"

        data_dict = pkl.load(open(data_dict_path, 'rb'))
        metadata = pkl.load(open(metadata_path, 'rb'))

        return data_dict, metadata

    def get_dataset_data_dict(self):
        data_dict = {}
        metadata = {}
        # need to implement the function to get the dataset from the root dir

        return data_dict, metadata

    def get_participants_ids(self):
        return self.data_dict.keys()

    def get_participant_data_dict(self, participant_id):
        return self.data_dict[participant_id]

    def initialize_x_y(self, **kwargs):
        return np.array([]), np.array([])


class FACED(PhysioDataset):
    # "Angry": [0, 1, 2],
    # "Disgust": [3, 4, 5],
    # "Fear": [6, 7, 8],
    # "Sadness": [9, 10, 11],
    # "Amusement": [16, 17, 18],
    # "Inspiring": [19, 20, 21],
    # "Joy": [22, 23, 24],
    # "Tenderness": [25, 26, 27]

    y1_label_mapping = {
        0: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 1,
        6: 2,
        7: 2,
        8: 2,
        9: 3,
        10: 3,
        11: 3,
        16: 4,
        17: 4,
        18: 4,
        19: 5,
        20: 5,
        21: 5,
        22: 6,
        23: 6,
        24: 6,
        25: 7,
        26: 7,
        27: 7
    }
    # map to negative and positive
    y2_label_mapping = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        11: 0,
        16: 1,
        17: 1,
        18: 1,
        19: 1,
        20: 1,
        21: 1,
        22: 1,
        23: 1,
        24: 1,
        25: 1,
        26: 1,
        27: 1

    }



    def __init__(self, dataset_name, load_from_exist, raw_data_root_dir=None, processed_data_root_dir=None, save_dataset = True, **kwargs):
        super().__init__(dataset_name, load_from_exist, raw_data_root_dir, processed_data_root_dir, save_dataset)

        self.metadata = kwargs
        self.data_dict, self.metadata = self.load_data_dict()

    def get_dataset_data_dict(self):

        data_dict = {}
        video_labels = self.metadata['video_labels']
        neutral_video_indices = video_labels['Neutral']
        fs  = self.metadata['preprocessing_config']['resample_fs']
        metadata = self.metadata
        foldPaths = os.path.join(self.raw_data_dir, 'Recording_info.csv')
        subs = pd.read_csv(foldPaths, low_memory=False)['sub']


        for sub in subs:

            sub_path = os.path.join(self.raw_data_dir, 'Processed_data',sub + '.pkl')
            assert os.path.exists(sub_path), f"Path {sub_path} does not exist"
            data = pd.read_pickle(sub_path)

            sub_x = data
            sub_y = np.array(arange(len(data)))

            # remove the neutral videos labels from the sub_y and sub_x
            # find the indices of the neutral videos
            mask = np.isin(sub_y, neutral_video_indices)
            sub_x = sub_x[~mask]
            sub_y = sub_y[~mask]

            # map sub_y to the 8 classes
            sub_y1 = np.vectorize(self.y1_label_mapping.get)(sub_y)
            sub_y2 = np.vectorize(self.y2_label_mapping.get)(sub_y)

            #  'data' is your EEG data array with shape (24, 32, 7500)
            num_trials, num_channels, num_time_points = sub_x.shape
            window_size = 250  # Size of each window in time points
            num_windows = num_time_points // window_size  # Total number of windows per trial

            # Reshape and transpose the data
            sub_x = sub_x.reshape(num_trials, num_channels, num_windows, window_size)
            sub_x = sub_x.transpose(0, 2, 1, 3)

            sub_x = sub_x.reshape(num_trials * num_windows, num_channels, window_size)


            # repeat the labels for each window
            sub_y1 = np.repeat(sub_y1, num_windows)
            sub_y2 = np.repeat(sub_y2, num_windows)


            # have another layer for dictionary for eeg
            data_dict[sub] = {'eeg': {'run-1': {'x': sub_x, 'y1': sub_y1, 'y2': sub_y2}}}

        return data_dict, metadata


    def initialize_x_y(self):
        # need to split the data into smaller windows, this will depend on how the model is implemented and the classification task
        return np.array([]), np.array([])


class AuditoryOddballDelorme2020(PhysioDataset):

    y_label_mapping = {
        2: 0,
        8: 1
    }

    def __init__(self, dataset_name, load_from_exist, raw_data_root_dir=None, processed_data_root_dir=None, save_dataset = True, **kwargs):
        super().__init__(dataset_name, load_from_exist, raw_data_root_dir, processed_data_root_dir, save_dataset)

        self.metadata = kwargs
        self.data_dic, self.metadata = self.load_data_dict()

    def get_dataset_data_dict(self):
        # need to implement the function to get the dataset from the root dir
        data_dict = {}
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

        event_ids = metadata['event_ids']

        num_subjects, sub_dirs = count_dirs_starting_with(self.raw_data_dir, 'sub-')

        for sub_num, sub_dir in enumerate(sub_dirs):

            subject = sub_dir.split("-")[1]
            sub_path = os.path.join(bids_root, sub_dir)
            bids_path = BIDSPath(root=bids_root, datatype=datatype)
            # get the number of runs for this subject
            data_path = os.path.join(sub_path, datatype)
            num_runs, run_ids = count_bids_runs(data_path)

            sub_data_dict = {'eeg': {}}

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
                    remapped_event_id = {key: event_ids[key] for key in event_id.keys()}
                except KeyError as e:
                    print(f"KeyError: {e}")
                    print(f"event_id: {event_id}")
                    print(f"master_event_id: {event_ids}")
                    raise
                # change the local event id to the master event id which is same as the 'label' in the metadata
                for i in range(len(events)):
                    # If the original event in the matrix matches the old event_id,
                    # replace it with the new event_id from the master_event_id dictionary
                    event_find = False
                    for events_key, events_value in event_id.items():
                        if events[i, 2] == events_value:
                            events[i, 2] = event_ids[events_key]
                            event_find = True
                            break
                    if not event_find:  # the remapped event_id is not found
                        print(f"Event not found in event_id: {events[i, 2]}")
                        # we need exit the program with exception
                        raise ValueError("Event not found in event_id")

                epochs = mne.Epochs(raw, events, remapped_event_id, tmin=epoch_t_min, tmax=epoch_t_max,
                                    baseline=baseline, preload=True)


                # uncomment this to visualize the epochs

                # visualize_epochs_channels(epochs,
                #                           event_groups={"stimulus/standard": 2, "stimulus/oddball_with_reponse": 8},
                #                           colors={"stimulus/standard": "blue", "stimulus/oddball_with_reponse": "red"},
                #                           picks=['Fz', 'Cz', 'Pz', 'Oz'],
                #                           tmin_vis=-0.1, tmax_vis=0.8,
                #                           title=f'Auditory Oddball Delorme 2020 - Participant',
                #                           out_dir=None, verbose='INFO', fig_size=(12.8, 7.2))

                run_data = epochs.get_data(picks='eeg')
                run_labels = epochs.events[:, -1]

                # only keep the 2 and 8 labels
                run_data = run_data[np.isin(run_labels, [2, 8])]
                run_labels = run_labels[np.isin(run_labels, [2, 8])]
                # remap the labels
                run_labels = np.vectorize(self.y_label_mapping.get)(run_labels)


                sub_data_dict['eeg']['run-' + run_id] = {'x': run_data, 'y': run_labels}


            data_dict['sub' + subject] = sub_data_dict

        metadata['ch_names'] = montage.ch_names

        return data_dict, metadata

    def initialize_x_y(self):
        pass

class BCICompetitionIV2a(PhysioDataset):
    """

    This class is used to get the BCICIVA dataset
    Dataset location: https://www.bbci.de/competition/iv/
    Description: https://www.bbci.de/competition/iv/desc_2a.pdf

    """

    y_label_mapping = {
        1:0,
        2:1,
        3:2,
        4:3,
    }

    def __init__(self, dataset_name, load_from_exist, raw_data_root_dir=None, processed_data_root_dir=None, save_dataset = True, **kwargs):
        super().__init__(dataset_name, load_from_exist, raw_data_root_dir, processed_data_root_dir, save_dataset)

        self.metadata = kwargs
        self.data_dict, self.metadata = self.load_data_dict()


    # note: there are two runs, the first run will be used as the training data, and the second run will be used as the testing data
    def get_dataset_data_dict(self):

        data_dict = {}

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

            sub_data_dict = {'T': {}, 'E': {}}

            for session in sessions:

                session_data_dict = {}

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

                    # filter to [4, 40] Hz
                    raw = bandpass_filter_raw(raw, 4, 40, picks='eeg')

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

                    # remap the labels
                    labels = np.vectorize(self.y_label_mapping.get)(labels)

                    session_data_dict['run-' + str(run)] = {'x': data, 'y': labels}

                sub_data_dict[session]['eeg'] = session_data_dict

            data_dict[subject] = sub_data_dict

        return data_dict, metadata

    def initialize_x_y(self):
        print("initialize_x_y")
        pass


if __name__ == '__main__':
    # try get FACED dataset

    raw_data_root_dir = 'C:/Dataset/raw'
    processed_data_root_dir = 'C:/Dataset/processed'

    # read the json file
    with open('eeg_dataset_config.json') as f:
        config = json.load(f)

    # dataset_name = 'FACED'
    # dataset_metadata = config['FACED']
    # faced = FACED(dataset_name, load_from_exist=False, raw_data_root_dir=raw_data_root_dir, processed_data_root_dir=processed_data_root_dir, save_dataset=True, **dataset_metadata)
    #
    #
    #
    #
    # dataset_name = 'AuditoryOddballDelorme2020'
    # dataset_metadata = config['AuditoryOddballDelorme2020']
    # auditory = AuditoryOddballDelorme2020(dataset_name, load_from_exist=False, raw_data_root_dir=raw_data_root_dir, processed_data_root_dir=processed_data_root_dir, save_dataset=True, **dataset_metadata)


    dataset_name = 'BCICompetitionIV2a'
    dataset_metadata = config['BCICompetitionIV2a']
    bci = BCICompetitionIV2a(dataset_name, load_from_exist=False, raw_data_root_dir=raw_data_root_dir, processed_data_root_dir=processed_data_root_dir, save_dataset=True, **dataset_metadata)

    print("Finished")