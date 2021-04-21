import matplotlib.pyplot as plt
import seaborn as sns

from pytsal.internal.entity import TimeSeries
from pytsal.internal.utils.helpers import get_logger
from pytsal.visualization.base_visualization import VisualizeContainer

LOG = get_logger(__name__)


class EDAVisualizer(VisualizeContainer):
    """
        Time series EDA Visualizer
    """

    def __init__(self, ts: TimeSeries, plotter='matplotlib', init_seaborn=True):
        super().__init__(ts, plotter, 'EDA', init_seaborn)

    def __set_axis_labels(self):
        plt.xlabel('DateTime')
        plt.ylabel(self.ts_prop['target'])

    def plot(self):
        plt.plot(self.ts)
        plt.title(self.ts_prop['name'])
        self.__set_axis_labels()
        plt.show()

    def decompose(self, model: str = 'add'):
        """
        The method of seasonal decompose splits the time series `Y[t]` into respective temporal components.

            1. Additive model

                In this model the summation of the temporal components form the time series
                Y[t] = T[t] + S[t] + L[t] + r

            2. Multiplicative model

                The product of the temporal components form the time series
                Y[t] = T[t] * S[t] * L[t] * r

        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        decompose_result = seasonal_decompose(self.ts, model=model)
        decompose_result.plot().show()

    def seasonal_plot(self):
        """
        A seasonal plot is similar to a time plot except that the data are plotted against the individual **seasons**
        in which the data were observed.

        Observations that can be made from seasonal plot are as follows:
            1. Whether the time series has seasonality or not
            2. Nature of seasonality (additive or multiplicative). If the nature is multiplicative then the consecutive vertical lines would have increasing amplitude
            3. Determine pattern changes
        """
        fig, ax = plt.subplots(figsize=(15, 6))

        # TODO: Fix this index.month and index.year later it might break for other cases
        sns.lineplot(x=self.ts.index.month, y=self.ts.values, hue=self.ts.index.year, legend="full")
        ax.legend(loc='right')
        ax.set_title('Seasonal plot', fontsize=20)
        self.__set_axis_labels()
        plt.show()

    def box_plot(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

        sns.boxplot(x=self.ts.index.year, y=self.ts.values, ax=ax[0])

        ax[0].set_title('Year-wise Box Plot', fontsize=20)
        ax[0].set_xlabel('Year')
        plt.ylabel(self.ts_prop['target'])
        fig.autofmt_xdate()

        sns.boxplot(x=self.ts.index.month, y=self.ts.values, ax=ax[1])
        ax[1].set_title('Month-wise Box Plot', fontsize=20)
        ax[1].set_xlabel('Month')
        plt.ylabel(self.ts_prop['target'])
        plt.show()

    def acf_and_pacf_plot(self):
        from statsmodels.graphics import tsaplots
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        f = tsaplots.plot_acf(self.ts.values, ax=ax[0])
        f = tsaplots.plot_pacf(self.ts.values, ax=ax[1])
        plt.show()

    def test_stationarity(self, window=12, cutoff=0.01):
        from statsmodels.tsa.stattools import adfuller

        rolmean = self.ts.rolling(window).mean()
        rolstd = self.ts.rolling(window).std()

        fig = plt.figure(figsize=(12, 8))
        orig = plt.plot(self.ts, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')

        plt.title('Stationarity Test')
        plt.show()

        # Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(self.ts, autolag='AIC', maxlag=20)

        import pandas as pd
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput[f'Critical Value {key}'] = value

        pvalue = dftest[1]
        if pvalue < cutoff:
            text = 'p-value = %.4f. The series is likely stationary.' % pvalue
        else:
            text = 'p-value = %.4f. The series is likely non-stationary.' % pvalue
        print(text)

        print(dfoutput)

    def summary_plot(self):
        """
        Plots all the necessary EDA plots on a single function call
        """
        LOG.info(f'{self.__class__.__name__} initialized')
        self.plot()
        LOG.info('Constructed Time plot')
        self.decompose()
        LOG.info('Constructed decompose plot')
        self.seasonal_plot()
        LOG.info('Constructed seasonal plot')
        self.box_plot()
        LOG.info('Constructed box plot')
        self.acf_and_pacf_plot()
        LOG.info('Constructed acf and pacf plot')
        self.test_stationarity()
        LOG.info('Performed stationary test plot')
        LOG.info(f'{self.__class__.__name__} completed')
