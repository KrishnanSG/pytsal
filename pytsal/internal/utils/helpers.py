import logging
import os
from math import ceil

from pytsal.internal.entity import *

logging.root.setLevel(logging.INFO)

# The following line sets the root logger level as well:
logging.basicConfig(level=logging.INFO)


def get_logger(name):
    return logging.getLogger(name)


def get_relative_path_from_root(filename: str):
    return os.sep.join([__file__[:-17].replace('/', os.sep), 'datasets', filename])


def get_freq(seconds: int):
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24
    freq = 'D'
    if days >= 31:
        freq = 'MS'
    elif 355 <= days:
        freq = str(int(days / 31)) + 'M'
    elif days > 365:
        freq = str(int(days / 365)) + 'Y'
    return freq


def split_into_train_test(ts: TimeSeries, split_ratio=0.7):
    size = ts.data.size
    train_size = ceil(size * split_ratio)
    train_data = ts.data[:train_size]
    test_data = ts.data[train_size:]
    print(f'"{ts.name}" dataset split with train size: {train_size} test size: {size - train_size}')
    splits = (
        TrainTS(train_data, ts.name + ' TrainSet', ts.target, freq=ts.freq),
        TestTS(test_data, ts.name + ' TestSet', ts.target, freq=ts.freq)
    )
    return splits
