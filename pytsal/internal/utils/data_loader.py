import pandas as pd


def get_freq(seconds: int):
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24
    freq = 'D'
    if days >= 31:
        freq = 'M'
    elif 355 <= days:
        freq = str(int(days / 31)) + 'M'
    elif days > 365:
        freq = str(int(days / 365)) + 'Y'
    return freq


def csv_loader(path: str, index: str = 'X', target: str = 'Y', freq=None) -> pd.Series:
    df = pd.read_csv(path)
    _datetime_index = pd.to_datetime(df[index])

    # Auto detect frequency
    seconds = (_datetime_index[1] - _datetime_index[0]).total_seconds()
    data = df[target].values

    # Create new pandas Series
    time_series = pd.Series(data=data, index=_datetime_index, name=target)
    _freq = get_freq(seconds) if freq is not None else freq

    # Set frequency
    time_series.asfreq(freq=_freq)

    return time_series
