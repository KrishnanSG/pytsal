import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import HoltWintersResults

from pytsal.internal.entity import TimeSeries
from pytsal.internal.utils.helpers import get_logger

LOG = get_logger(__name__)


def brutlag_algorithm(ts: TimeSeries, model: HoltWintersResults, tolerance: int = 0.5, period=12, sf=2,
                      gamma=0.3684211):
    LOG.info('Performing anomaly detection ...')

    # period - seasonal_period of the given time series
    # gamma - the seasonality component (0.2 -0.5)
    # sf - brutlag scaling factor for the confidence bands. (2-3)
    ub = []  # upper bound or upper confidence band
    lb = []  # lower bound or lower confidence band
    prediction = model.predict(ts.start, ts.end)

    difference_array = []
    dt = []
    difference_table = {
        "actual": ts.data, "predicted": prediction, "difference": difference_array, "UB": ub, "LB": lb}

    """Calculation of confidence bands using brutlag algorithm"""
    for i in range(len(prediction)):
        diff = ts.data[i] - prediction[i]
        if i < period:
            dt.append(gamma * abs(diff))
        else:
            dt.append(gamma * abs(diff) + (1 - gamma) * dt[i - period])
        difference_array.append(diff)
        ub.append(prediction[i] + sf * dt[i])
        lb.append(prediction[i] - sf * dt[i])

    print("\nDifference between actual and predicted\n")

    difference = pd.DataFrame(difference_table)
    print(difference)

    """Classification of data points as either normal or anomaly"""
    normal = []
    normal_date = []
    anomaly = []
    anomaly_date = []
    actual = []

    for i in range(len(ts.data.index)):
        if (ub[i] <= ts.data[i] or lb[i] >= ts.data[i]) and i > period and abs(ts.data[i] - prediction[i]) > tolerance:
            anomaly_date.append(ts.data.index[i])
            anomaly.append(ts.data[i])
            actual.append(prediction[i])
        else:
            normal_date.append(ts.data.index[i])
            normal.append(ts.data[i])

    anomaly = pd.DataFrame({"date": anomaly_date, "observed": anomaly, 'expected': actual})
    anomaly.set_index('date', inplace=True)
    normal = pd.DataFrame({"date": normal_date, "value": normal})
    normal.set_index('date', inplace=True)

    print("\nThe data points classified as anomaly\n")
    print(anomaly)

    """
    Plotting the data points after classification as anomaly/normal.
    Data points classified as anomaly are represented in red and normal in green.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(normal.index, normal, 'o', color='green')
    plt.plot(anomaly.index, anomaly[['observed']], 'o', color='red')

    # Plotting brutlag confidence bands
    plt.plot(ts.data.index, ub, linestyle='--', color='grey')
    plt.plot(ts.data.index, lb, linestyle='--', color='grey')

    # Formatting the graph
    plt.legend(['Normal', 'Anomaly', 'Upper Bound', 'Lower Bound'])
    plt.gcf().autofmt_xdate()
    plt.title(ts.name)
    plt.xlabel('Datetime')
    plt.ylabel(ts.target)
    plt.show()
    return anomaly
