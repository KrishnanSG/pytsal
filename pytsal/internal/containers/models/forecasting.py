from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from statsmodels.tsa.holtwinters import HoltWintersResults

from pytsal.internal.containers.models.base_model import ModelContainer
from pytsal.internal.entity import TrainTS, TestTS, TimeSeries
from pytsal.visualization.validation import find_mape, find_mae


class Forecasting(ModelContainer, ABC):

    def __init__(self, id: str, name: str, model: Any, args: Dict[str, Any] = None):
        self.model_args = args
        super().__init__(id, name, model, args=args)

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def forecast(self, *args, **kwargs):
        pass

    @abstractmethod
    def score(self, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def get_tunable():
        pass


class HoltWinter(Forecasting):

    def __init__(self, model_args: Dict = None):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        super().__init__('holtwinter', 'Holt Winter', ExponentialSmoothing, args=model_args)

    @staticmethod
    def __find_model_args(ts: TrainTS):
        from statsmodels.tsa.seasonal import seasonal_decompose
        decompose_result = seasonal_decompose(ts.data)
        trend_values = decompose_result.trend.values

        # Find seasonal period and test for trend
        period = 0
        for i in trend_values:
            try:
                int(i)
                break
            except ValueError:
                period += 1
        trend = 'additive' if (trend_values[period + 1] - trend_values[-period - 1]) != 0 else None
        trend_nature = None
        if trend:
            trend_nature = 'Increasing' if (trend_values[period + 1] - trend_values[-period - 1]) < 0 else 'Decreasing'

        # Test seasonality
        seasonal_period = period * 2
        seasonal_values = decompose_result.seasonal.values
        seasonal = None
        if seasonal_values[0] == seasonal_values[seasonal_period] and seasonal_values[0] != seasonal_values[period]:
            seasonal = 'mul'

        model_args = {
            'trend': trend,
            'trend_nature': trend_nature,
            'seasonal': seasonal,
            'seasonal_period': seasonal_period,
        }
        return model_args

    def fit(self, ts: Union[TrainTS, TimeSeries], args: Dict = {}) -> HoltWintersResults:
        if self.model_args is None:
            self.model_args = self.__find_model_args(ts)
        return self.class_def(ts.data, trend=self.model_args['trend'], seasonal=self.model_args['seasonal']).fit(**args)

    def forecast(self, model: HoltWintersResults, n_steps):
        return model.forecast(n_steps)

    def predict(self, model: HoltWintersResults, start, end):
        return model.predict(start, end)

    def score(self, ts: TestTS, model: HoltWintersResults):
        predicted_values = model.predict(ts.start, ts.end)
        return {
            'MAE': find_mae(ts.data.values, predicted_values.values),
            "MAPE": find_mape(ts.data.values, predicted_values.values),
            'AIC': model.aic,
            'AICC': model.aicc,
            'BIC': model.bic,
            'SSE': model.sse
        }

    @staticmethod
    def get_tunable():
        return [
            dict(trend=None, seasonal=None),
            dict(trend='add', seasonal=None),
            dict(trend=None, seasonal='add'),
            dict(trend='mul', seasonal=None),
            dict(trend=None, seasonal='mul'),
            dict(trend='add', seasonal='add'),
            dict(trend='add', seasonal='mul'),
            dict(trend='mul', seasonal='add'),
            dict(trend='mul', seasonal='mul'),
        ]


class SARIMAX(Forecasting):
    def fit(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass

    def forecast(self, *args, **kwargs):
        pass

    def score(self, ts: TestTS, model):
        predicted_values = model.predict(ts.start, ts.end)
        return {
            'MAE': find_mae(ts.data.values, predicted_values.values),
            "MAPE": find_mape(ts.data.values, predicted_values.values),
            'AIC': model.aic,
            'AICC': model.aicc,
            'BIC': model.bic,
            'SSE': model.sse
        }

    @staticmethod
    def get_tunable():
        pass


MODELS = {
    'holtwinter': HoltWinter,

}

if __name__ == '__main__':
    ARIMA().fit()