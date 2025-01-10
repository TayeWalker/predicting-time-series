import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast import SampleForecast
from gluonts.dataset.util import to_pandas

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

import json
import os
import pickle
from datetime import datetime



config_path = "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

parameters = config
# Set device
device = 'cpu'
SIZE = "large"
CTX = parameters["CTX"]
PDT = parameters["PDT"]
PSZ = parameters['PSZ']
BSZ = parameters["BSZ"]
data_set = parameters["data_set"]
samples = parameters['NSP']

# Read the CSV file
df = pd.read_csv(data_set, parse_dates=['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Ensure the index has a weekly frequency
if df.index.freq is None:
    df = df.asfreq('W')

print("DataFrame index frequency:", df.index.freq)
print("DataFrame columns:", df.columns.tolist())

# Create a ListDataset for testing
test_data = ListDataset(
    [
        {
            FieldName.START: df.index[0], # There is no x variable we just have a start time and step forward by freq
            FieldName.TARGET: df[region].values,
            FieldName.ITEM_ID: region
        }
        for region in df.columns
    ],
    freq='W'  # Weekly frequency
)

# Initialize the model
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained(
        f"Salesforce/moirai-1.0-R-{SIZE}",
        map_location=device
    ),
    prediction_length=PDT,
    context_length=CTX,
    patch_size=PSZ,
    num_samples=samples,
    target_dim=1,  # Since we are forecasting one region at a time
    feat_dynamic_real_dim=0, 
    past_feat_dynamic_real_dim=0, 
)


# Create predictor
predictor = model.create_predictor(batch_size=BSZ, device=torch.device(device))

# Generate forecasts using make_evaluation_predictions
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data,
    predictor=predictor,
    num_samples=samples, 
    #num_workers=0,
)

# Collect forecasts and time series
forecasts = list(forecast_it)
tss = list(ts_it)

for i, forecast in enumerate(forecasts):
    print(f"Forecast {i}: samples shape = {forecast.samples.shape}")


predictions = forecasts[0].samples
np.save("forecast", predictions)
