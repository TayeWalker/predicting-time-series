import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') #agg is a backend for non-interactive enviroments
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_absolute_error

from gluonts.evaluation.backtest import make_evaluation_predictions #function to generate forecasts
from gluonts.dataset.field_names import FieldName #constants representing field names in data entries.
from gluonts.model.forecast import SampleForecast #forecast object containing samples of future values.

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule #model

import json
import os
from datetime import datetime

# Custom dataset class for handling the multivariate data. 
# It seems clunky to use an iterator becuase we are only yielding one value, but I believe that's the expected form.
class CustomTimeSeriesDataset:
    def __init__(self, df, prediction_length, freq):
        self.df = df
        self.prediction_length = prediction_length
        self.freq = freq
    
    def __iter__(self):
        yield {
            FieldName.START: self.df.index[0],  # Gives the start date. There is no x variable we just have a start time and step forward by freq
            FieldName.TARGET: self.df.values.T,  # 2D shape: (variables, time steps). This field contains almost all the information
            FieldName.ITEM_ID: "multivariate_series" 
        }
    def __len__(self):
        return 1 #We are only yielding one value so this is true

config_path = "config.json"
with open(config_path, "r") as f:
    config = json.load(f)
parameters = config

data_set = parameters["data_set"]
#Important Parameters
device = 'cpu'
SIZE = "large"
CTX = parameters["CTX"] # Prediction length. Number of points we are going to predict.
PDT = parameters["PDT"] # Context length. This is the length of the window from which we predict a single point.
BSZ = parameters["BSZ"] # Batch size; smaller sizes improve computation time 
PSZ = parameters['PSZ']
samples = parameters['NSP'] # Times that we sample from the distribution

# Load Data
df = pd.read_csv(data_set, parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Convert index to PeriodIndex with weekly frequency for compatibility with GluonTS
df.index = df.index.to_period("W")


# Drop NaNs and set to float
df = df.dropna().astype(float)

# Custom dataset initialization
test_data = CustomTimeSeriesDataset(df=df, prediction_length=PDT, freq="W")

# Initialize the model
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained(
        f"Salesforce/moirai-1.0-R-{SIZE}",
        map_location=device
    ),
    prediction_length=PDT,
    context_length=CTX,
    patch_size=PSZ, 
    num_samples=samples, # Times that we sample from the distribution
    target_dim=df.shape[1], #Number of variables
    feat_dynamic_real_dim=0, 
    past_feat_dynamic_real_dim=0, #these last two parameters would be a place to add aditional features
)

# This our model into a GlounTS predictor. The predictor handles the model's inference process, including batching and device allocation.
predictor = model.create_predictor(batch_size=BSZ, device=torch.device(device))

# forecast_it is an iterator over the forecast object
#ts_it is an itorator over the target (ground truth tiem series)
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data,
    predictor=predictor,
    num_samples=samples
)

# Convert forcasts to lists
forecasts = list(forecast_it)
tss = list(ts_it)


predictions = forecasts[0].samples

np.save(f"forecast_{data_set[:-4]}", predictions)
