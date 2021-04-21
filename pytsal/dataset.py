import pandas as pd

from pytsal.internal.entity import TimeSeries
from pytsal.internal.utils.data_loader import csv_loader
from pytsal.internal.utils.helpers import get_relative_path_from_root


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
    name = 'Monthly totals of international airline passengers (1949 to 1960)'
    target = 'Number of airline passengers'
    data = csv_loader(path, index='Date', target=target)
    return TimeSeries(data, name, target, freq=data.index.freqstr)


def load_airline_with_anomaly() -> TimeSeries:
    """
        Load the airline univariate time series dataset with random anomaly points
    """
    filename = 'airline_with_anomaly.csv'
    path = get_relative_path_from_root(filename)
    name = 'Monthly totals of international airline passengers (1949 to 1960)'
    target = 'Number of airline passengers'
    data = csv_loader(path, index='Date', target=target)
    return TimeSeries(data, name, target, freq=data.index.freqstr)
