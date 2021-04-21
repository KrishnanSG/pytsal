from abc import ABC, abstractmethod

from pytsal.internal.entity import TimeSeries


class VisualizeContainer(ABC):
    def __init__(self, ts: TimeSeries, plotter, name: str, init_seaborn: bool = True):
        self.init_seaborn = init_seaborn
        self.ts = ts.data
        self.ts_prop = ts.__dict__
        self.plotter = plotter
        self.name = name

        if init_seaborn:
            self.__init_seaborn()

    @staticmethod
    def __init_seaborn():
        import seaborn as sns
        sns.set()

    @abstractmethod
    def summary_plot(self):
        pass
