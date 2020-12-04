import random
import numpy as np
import pandas as pd
import torch


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_data(filename):
    data = pd.read_csv(filename, sep='|', header=None)
    data = data.drop(columns=[1, 2]).dropna()
    data.columns = ['path']
    return data


def split_data(data, valid_ratio=0.1):
    valid_size = int(valid_ratio * data.shape[0])
    valid_index = np.random.choice(data.index, size=valid_size, replace=False)
    train_index = np.setdiff1d(data.index, valid_index)

    train_data = data.loc[train_index].reset_index(drop=True)
    valid_data = data.loc[valid_index].reset_index(drop=True)

    return train_data, valid_data
