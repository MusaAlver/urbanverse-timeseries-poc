"""Central experiment configuration for the UrbanVerse time-series PoC."""

DATASET_NAME = "METR-LA"
PRIMARY_SENSOR = "773062"
GENERALIZATION_SENSOR = "717608"
CONTEXT_LENGTH = 96
FORECAST_HORIZON = 24
SAMPLING_MINUTES = 5

# Required evaluation metrics from the research brief.
REQUIRED_METRICS = ("MAE", "RMSE")
