"""
Holt Winter Forecasting Example
    In this example we will look into how to create holt winters model and save the model in less than 4 steps.
"""

from pytsal.dataset import *
from pytsal.forecasting import *

# 1. Load the time series dataset
ts = load_airline()

# 2. Setup the the experiment/model
model = setup(ts, 'holtwinter', eda=True, validation=True, find_best_model=True, plot_model_comparison=True)

# 3. Finalize the model for production. Finalizing the model trains it on the complete data.
trained_model = finalize(ts, model)

# 4. Save the model
# save_model(trained_model)
