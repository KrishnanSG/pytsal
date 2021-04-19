import pandas as pd

from pytsal.internal.utils.data_loader import csv_loader
from pytsal.internal.utils.helpers import get_relative_path_from_root


class TimeSeries:
    def __init__(self, ts: pd.Series, name: str, target: str, freq=None):
        self.data = ts
        self.name = name
        self.freq = freq
        self.target = target

    def __str__(self):
        return str(self.data)


def load_airline() -> TimeSeries:
    """
        Load the airline univariate time series dataset.

        Returns
        -------
        y : pd.Series
         Time series

        Details
        -------
        The classic Box & Jenkins airline data. Monthly totals of international
        airline passengers, 1949 to 1960.
        Dimensionality:     univariate
        Series length:      144
        Frequency:          Monthly
    """
    filename = 'airline.csv'
    path = get_relative_path_from_root(filename)
    name = 'Monthly totals of international airline passengers, 1949 to 1960'
    target = 'Number of airline passengers'
    data = csv_loader(path, index='Date', target=target)
    return TimeSeries(data, name, target, freq=data.index.freqstr)
