import os
import pyedflib
import numpy as np
from torch.utils.data import Dataset

from .preprocess import low_pass_filtering, high_pass_filtering, StandardScaler


ECG_MIN = -0.06
ECG_MAX = 0.12


class EdfDataset(Dataset):
    """Torch dataset for edf file.

    Load dataset from a single edf file.
    """

    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        self.data_path = data_path
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self._read_data()

    def _read_data(self):
        self.scaler = StandardScaler()
        self.ecg = []

        with pyedflib.EdfReader(os.path.join(self.root_path,
                                             self.data_path)) as f:
            ecg = f.readSignal(0)
            ecg = ecg[::2]
            ecg = low_pass_filtering(ecg)
            ecg = high_pass_filtering(ecg)
            ecg[ecg < ECG_MIN] = ECG_MIN
            ecg[ecg > ECG_MAX] = ECG_MAX
            self.ecg = ecg

        total_len = len(self.ecg)
        num_train = int(total_len * 0.7)
        num_test = int(total_len * 0.2)
        num_vali = total_len - num_train - num_test
        border_starts = [0, num_train-self.seq_len,
                         total_len-num_test-self.seq_len]
        border_ends = [num_train, num_train+num_vali, total_len]
        border_start = border_starts[self.set_type]
        border_end = border_ends[self.set_type]
        ecg = self.ecg[border_start:border_end]

        train_data = self.ecg[border_starts[0]:border_ends[0]]
        self.scaler.fit(train_data)
        ecg = self.scaler.transform(ecg)
        self.ecg = np.expand_dims(ecg, axis=1)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.ecg[s_begin:s_end]
        seq_y = self.ecg[r_begin:r_end]
        return seq_x, seq_y, np.arange(s_begin, s_end), np.arange(r_begin, r_end)

    def __len__(self):
        return len(self.ecg) - self.seq_len - self.pred_len + 1
