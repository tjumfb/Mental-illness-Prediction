# sys
import pickle

import numpy as np
# torch
import torch
from torch.utils import data


class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 debug=False):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.load_data()

    def load_data(self):
        # data: N * 768
        # label: N * 1

        # load label
        with open(self.label_path, 'rb') as f:
           self.label = pickle.load(f)
        with open(self.data_path, 'rb') as f:
           self.data = pickle.load(f)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = torch.from_numpy(self.data[index])
        label = torch.tensor(self.label[index])

        return data_numpy, label