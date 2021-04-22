from datetime import datetime
from typing import Any

import pandas as pd

from pytsal.internal.containers.models.forecasting import MODELS, Forecasting
from pytsal.internal.entity import TimeSeries, TrainTS, TestTS
from pytsal.internal.utils.helpers import get_logger, split_into_train_test
from pytsal.visualization.eda import EDAVisualizer
from pytsal.visualization.validation import ValidationVisualizer

LOG = get_logger(__name__)


def setup(
        ts: TimeSeries,
        model_name: str,
        override_model=None,
        eda: bool = True,
        validation: bool = True,
        find_best_model: bool = True,
        validation_metric_name: str = 'MAE',
        plot_model_comparison=True
):
    import warnings
    warnings.filterwarnings("ignore")

    LOG.info(f'Experiment started @ {datetime.now()}')
    # Load time series
    LOG.info('Loading time series ...')
    print('\n--- Time series summary ---\n')
    print(ts.summary())

    # Create train test set
    LOG.info('Creating train test data')
    train, test = split_into_train_test(ts, split_ratio=0.8)

    if eda:
        LOG.info('Initializing Visualizer ...')
        eda = EDAVisualizer(ts)
        eda.summary_plot()

    # Create model
    if override_model is None:
        model = MODELS[model_name]()
    else:
        model = override_model

    if find_best_model and override_model is None:
        model, fit_model = tune_model(train, test, MODELS[model_name], metric_name=validation_metric_name,
                                      plot_comparison=plot_model_comparison)
    else:
        LOG.info('Initializing Model ...')
        fit_model = model.fit(train)

    print('--- Model Summary ---')
    print(model.model_args)

    # Validation
    if validation:
        viz = ValidationVisualizer(ts, train, test, fit_model)
        viz.summary_plot()

    LOG.info(f'Experiment end @ {datetime.now()}')

    return model


def tune_model(train: TrainTS, test: TestTS, model_class: Any, metric_name: str = 'MAE', plot_comparison: bool = True):
    LOG.info('Initialize model tuning ...')

    # Ignore erd party warning since most of them are deprecated warning and they add noise to the output
    import warnings
    warnings.filterwarnings("ignore")

    tunable = model_class.get_tunable()
    print(f'{len(tunable)} tunable params available for {model_class}')

    min_score = 1e5
    name = []
    args = []
    scores = []
    aicc = []

    best_model = None
    best_fit_model = None

    if plot_comparison:
        LOG.info('Initializing comparison plot ...')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.title('Model Comparison')
        legends = ['train', 'test']
        plt.plot(train.data)
        plt.plot(test.data, color='black')

    for param in tunable:

        model = model_class(model_args=param)
        fit_model = model.fit(train)
        aicc.append(round(fit_model.aicc, 2))
        score = model.score(test, fit_model)[metric_name]
        if score < min_score:
            best_model = model
            best_fit_model = fit_model
            min_score = score
        name.append(model_class.__name__)
        args.append(param)
        scores.append(score)

        if plot_comparison:
            plt.plot(model.predict(fit_model, test.start, test.end))
            legends.append(str(param))

    if plot_comparison:
        plt.legend(legends)
        plt.show()

    summary = pd.DataFrame({
        "model_name": name,
        "args": args,
        "score": scores,
        "aicc": aicc
    }).sort_values('score', ascending=True)

    LOG.info('Tuning completed')
    print('\n ###### TUNING SUMMARY #####\n')
    print(summary)
    print(f'\nBest model: {best_model.model_args} with score: {min_score}')

    return best_model, best_fit_model


def finalize(ts, model: Forecasting):
    LOG.info('Finalizing model (Training on complete data) ... ')
    return model.fit(ts)


def save_model(model, filename: str = 'trained_model.pytsal'):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    LOG.info(f'Model saved to {filename}')
    return 'model saved'


def load_model(filename: str = 'trained_model.pytsal'):
    import pickle
    with open(filename, 'rb') as f:
        LOG.info(f'Model loaded from {filename}')
        return pickle.load(f)
