from pytsal.anomaly import brutlag_algorithm
from pytsal.dataset import *
from pytsal.forecasting import *

ts = load_airline()
model = setup(ts, 'holtwinter')
trained_model = finalize(ts, model)
save_model(trained_model)

model = load_model()
ts_with_anomaly = load_airline_with_anomaly()
brutlag_algorithm(ts_with_anomaly, model, tolerance=50)
