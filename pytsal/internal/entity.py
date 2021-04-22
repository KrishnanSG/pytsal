import pandas as pd


class TimeSeries:
    def __init__(self, ts: pd.Series, name: str, target: str, freq=None, phase: str = 'Full'):
        self.data = ts
        self.name = name
        self.freq = freq
        self.target = target
        self.phase = phase
        self.start = self.data.index.values[0]
        self.end = self.data.index.values[-1]

    def __str__(self):
        return str(self.data)

    def summary(self):
        return pd.Series({
            'name': self.name,
            'freq': self.freq,
            'target': self.target,
            'type': 'Univariate',
            'phase': self.phase,
            'series_length': self.data.size,
            'start': self.start,
            'end': self.end
        })


class TrainTS(TimeSeries):
    def __init__(self, ts: pd.Series, name: str, target: str, freq=None):
        super().__init__(ts, name, target, freq=freq, phase='train')


class TestTS(TimeSeries):
    def __init__(self, ts: pd.Series, name: str, target: str, freq=None):
        super().__init__(ts, name, target, freq=freq, phase='test')
