from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from mne.io import read_raw_edf
from mne.viz import plot_topomap
from mne.channels import make_standard_montage
from mne import create_info

from utils import standard_1020


class SingleEDFDataset(Dataset):
    """Process a single EDF file and create a dataset."""

    def __init__(self, file_path, window_size=2.0, step_size=2.0, threshold_std=5, mask_percentage=0.1):
        """
        Initialize the SingleEDFDataset class.

        Parameters:
        - file_path (str): Path to the EDF file.
        - window_size (float): Window size in seconds.
        - step_size (float): Step size between windows in seconds.
        - threshold_std (float): Threshold for clipping based on standard deviation.
        - mask_percentage (float): Percentage of time to mask in the windows.
        """
        self.file_path = file_path
        self.window_size = window_size
        self.step_size = step_size
        self.threshold_std = threshold_std
        self.mask_percentage = mask_percentage

        self.data = None
        self.sfreq = None
        self.dataset = None
        self.channel_names = None
        self.original_data = None

        # fixed list of "important" channels:
        self.important_channels = ['FP1', 'FP2', 'F4', 'C3', 'O1', 'C4', 'P3', 'P4', 'F7', 'O2', 'F8', 'T3', 'T5', 'T4', 'T6', 'F3', 'CZ', 'FZ', 'PZ', 'T2', 'T1']

        self._load_and_preprocess()

    def _load_and_preprocess(self):
        """Load the EDF file and preprocess into windows."""
        try:
            raw = read_raw_edf(self.file_path, preload=True)

            raw.rename_channels({ch_name: ch_name[:4] if ch_name.startswith('EEG ') else ch_name for ch_name in raw.ch_names})

            # Check if all "important" channels are present
            available_channels = set(raw.ch_names)
            if not all(channel in available_channels for channel in self.important_channels):
                raise ValueError("Not all required channels are present in the file.")

            raw.pick_channels(self.important_channels)  # select only the "important" channels

            imp_ch_set = set(self.important_channels)
            reordered_channels = []
            for ch_name in standard_1020:
                if ch_name in imp_ch_set:
                    reordered_channels.append(ch_name)
            raw.reorder_channels(reordered_channels)

            # Preprocess data
            raw.filter(0.5, 40)
            raw.resample(200)

            self.sfreq = raw.info['sfreq']
            self.channel_names = raw.ch_names
            self.original_data = raw.get_data(units='uV')

            data_array = self.original_data.copy()  # work on a copy of the original data
            mean = np.mean(data_array)
            std = np.std(data_array)
            data_array = np.clip(data_array, mean - self.threshold_std * std, mean + self.threshold_std * std)

            rms = np.sqrt(np.sum(data_array ** 2))
            data_array = (data_array - np.mean(data_array, axis=1, keepdims=True)) / rms

            n_samples_window = int(self.window_size * self.sfreq)
            n_samples_step = int(self.step_size * self.sfreq)

            windows = [
                data_array[:, start:start + n_samples_window]
                for start in range(0, data_array.shape[1] - n_samples_window + 1, n_samples_step)
            ]

            self.dataset = torch.tensor(np.array(windows))
        except ValueError as ve:
            print(f"Skipping file {self.file_path}: {ve}")
            self.dataset = None
        except Exception as e:
            raise RuntimeError(f"Error loading or preprocessing the EDF file: {e}")

    def __len__(self):
        return len(self.dataset) if self.dataset is not None else 0

    def __getitem__(self, idx):
        if self.dataset is None:
            raise IndexError("Dataset is empty or not processed due to missing required channels.")
        window = self.dataset[idx].clone()
        # mask_start_idx = random.randint(0, int(window.shape[-1] * (1 - self.mask_percentage)))
        # mask_end_idx = mask_start_idx + int(window.shape[-1] * self.mask_percentage)
        # masked_window = window.clone()
        # masked_window[:, mask_start_idx:mask_end_idx] = 0
        # return masked_window, window, [mask_start_idx, mask_end_idx]
        return window

    def get_ch_names(self):
        return [x.upper() for x in self.important_channels]

    @property
    def feature_size(self):
        return self.dataset[0].shape


class EDFDataset(Dataset):
    """Integrate multiple EDF files into a dataset."""

    def __init__(self, file_paths, window_size=2.0, step_size=2.0, threshold_std=5, mask_percentage=0.1):
        """
        Initialize the EDFDataset class.

        Parameters:
        - file_paths (list of str): List of paths to EDF files.
        - window_size (float): Window size in seconds.
        - step_size (float): Step size between windows in seconds.
        - threshold_std (float): Threshold for clipping based on standard deviation.
        - mask_percentage (float): Percentage of time to mask in the windows.
        """
        self.datasets = []
        for directories in file_paths:
            it = iter(Path(directories).iterdir())
            for i in range(10):
                file_path = next(it)
                self.datasets.append(SingleEDFDataset(file_path, window_size, step_size, threshold_std, mask_percentage))
        self.dataset_lengths = [len(dataset) for dataset in self.datasets]
        self.total_length = sum(self.dataset_lengths)

    @property
    def feature_size(self):
        return self.datasets[0].feature_size

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        dataset_idx = 0
        while idx >= self.dataset_lengths[dataset_idx]:
            idx -= self.dataset_lengths[dataset_idx]
            dataset_idx += 1
        return self.datasets[dataset_idx][idx]

    def free(self):
        pass

    def get_ch_names(self):
        return self.datasets[0].get_ch_names()