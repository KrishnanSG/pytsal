import pandas as pd

from pytsal.internal.utils.helpers import get_freq


def csv_loader(path: str, index: str = 'X', target: str = 'Y', freq=None) -> pd.Series:
    df = pd.read_csv(path)
    data = df[target].values
    _datetime_index = pd.to_datetime(df[index])

    # Auto detect frequency
    seconds = (_datetime_index[1] - _datetime_index[0]).total_seconds()
    _freq = get_freq(seconds) if freq is None else freq

    # Create new pandas Series
    time_series = pd.Series(data=data, index=pd.DatetimeIndex(_datetime_index, freq=_freq), name=target)
    return time_series
