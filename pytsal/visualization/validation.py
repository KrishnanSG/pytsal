from typing import Any

import matplotlib.pyplot as plt

from pytsal.internal.entity import TimeSeries, TrainTS, TestTS
from pytsal.internal.utils.helpers import get_logger
from pytsal.visualization.base_visualization import VisualizeContainer

LOG = get_logger(__name__)


def find_mae(expected, observed):
    import numpy as np
    output_errors = np.average(np.abs(observed - expected), axis=0)
    return np.average(output_errors)


def find_mape(expected, observed):
    import numpy as np
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(observed - expected) / np.maximum(np.abs(expected), epsilon)
    output_errors = np.average(mape, axis=0)
    return np.average(output_errors)


class ValidationVisualizer(VisualizeContainer):
    """
        Time series EDA Visualizer
    """

    def __init__(self, ts: TimeSeries, train: TrainTS, test: TestTS, model: Any, plotter='matplotlib',
                 init_seaborn=True):
        self.train = train
        self.test = test
        self.model = model
        self.predicted = self.model.predict(self.test.start, self.test.end)
        super().__init__(ts, plotter, 'Validation', init_seaborn)

    def __set_axis_labels(self):
        plt.xlabel('DateTime')
        plt.ylabel(self.ts_prop['target'])

    def plot(self):
        plt.plot(self.train.data)
        plt.plot(self.test.data)
        plt.plot(self.predicted)
        plt.legend(['Train', 'Test', 'Predicted'])
        plt.title('Validation')
        self.__set_axis_labels()
        plt.show()

    def error_plot(self):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(self.test.data)
        predicted_values = self.model.predict(self.test.start, self.test.end)
        plt.plot(self.test.data.index, predicted_values)
        errors = self.test.data.values - predicted_values.values
        plt.plot(self.test.data.index, errors, color='red')
        plt.legend(['Test', 'Predicted', 'Error'])
        plt.title('Error Plot')
        self.__set_axis_labels()
        plt.show()
        metrics = {
            'MAE': find_mae(self.test.data.values, predicted_values.values),
            "MAPE": find_mape(self.test.data.values, predicted_values.values),
            'AIC': self.model.aic,
            'BIC': self.model.bic,
            'AICC': self.model.aicc,
        }
        import pandas as pd
        print(pd.Series(metrics, name='Model metrics'))

    def summary_plot(self):
        """
        Plots all the necessary EDA plots on a single function call
        """
        self.plot()
        self.error_plot()
