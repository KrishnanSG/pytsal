from pytsal.internal.utils.data_loader import csv_loader


def load_airline():
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
    filename = '..\\pytsal\\internal\\datasets\\airline.csv'
    return csv_loader(filename, index='Date', target='Number of airline passengers')
