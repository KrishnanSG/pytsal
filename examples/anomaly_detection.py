from pytsal.anomaly import brutlag_algorithm
from pytsal.dataset import load_airline_with_anomaly
from pytsal.forecasting import load_model

# 1. Load the dataset
ts_with_anomaly = load_airline_with_anomaly()

# 2. Load the load or create a new HoltWinter forecasting model
model = load_model()

# 3. brutlag algorithm finds and returns the anomaly points
anomaly_points = brutlag_algorithm(ts_with_anomaly, model, tolerance=50)

