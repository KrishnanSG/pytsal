"""
Holt Winter Anomaly detection Example
    In this example we will look into how to create holt winters model and build an anomaly detection model
    in less than 4 steps.
"""

if __name__ == '__main__':

    from pytsal import anomaly, forecasting
    from pytsal.dataset import *

    # 1. Load the dataset
    ts_with_anomaly = load_airline_with_anomaly()

    # 2. Forecasting model

    # 2.a Load existing forecasting model
    model = forecasting.load_model()

    # 2.b Create new model
    if model is None:
        ts = load_airline()
        model = forecasting.setup(ts, 'holtwinter', eda=False, validation=False, find_best_model=True,
                                  plot_model_comparison=False)
        trained_model = forecasting.finalize(ts, model)
        forecasting.save_model(trained_model)
        model = forecasting.load_model()

    # 3. brutlag algorithm finds and returns the anomaly points
    anomaly_points = anomaly.setup(ts_with_anomaly, model, 'brutlag')

    print(anomaly_points)
