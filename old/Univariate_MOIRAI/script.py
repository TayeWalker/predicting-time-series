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
from datetime import datetime



config_path = "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

parameters = config["parameters"]
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


predictions = [forecast.quantile(0.5) for forecast in forecasts]
np.save("forecast", predictions)


output_dir = f"Data:{data_set}, PDT:{PDT}, CTX: {CTX}, PSZ: {PSZ}, BSZ: {BSZ}, samples: {samples}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Plot the results for each region
for region, ts, forecast in zip(df.columns, tss, forecasts):
    print(f"Processing region: {region}")
    
    # If ts is a DataFrame, convert it to a Series
    if isinstance(ts, pd.DataFrame):
        ts = ts.iloc[:, 0]
    
    # Adjust the index if necessary
    if isinstance(ts.index, pd.PeriodIndex):
        ts.index = ts.index.to_timestamp()
    elif not isinstance(ts.index, pd.DatetimeIndex):
        ts.index = pd.to_datetime(ts.index)
    
    # Convert forecast start date to Timestamp
    if isinstance(forecast.start_date, pd.Period):
        forecast_start = forecast.start_date.to_timestamp()
    else:
        forecast_start = pd.to_datetime(forecast.start_date)
    
    # Determine past_data_end
    if forecast_start in ts.index:
        past_data_end = forecast_start - pd.DateOffset(weeks=1)
    else:
        past_data_end = ts.index[ts.index < forecast_start][-1]
    
    # Slice past_data
    past_data = ts[:past_data_end]
    past_data = past_data[-(CTX + PDT):]  # Limit to desired length
    
    # Extract true future data
    future_data = ts[past_data_end:]
    future_data = future_data[:PDT]  # Limit to prediction length
    
    # Print data details
    print(f"Past data length: {len(past_data)}")
    print(f"Future data length: {len(future_data)}")
    
    # Check if past_data is empty
    if len(past_data) == 0:
        print(f"No past data to plot for region {region}")
        continue
    
    # Create date range for the forecast
    forecast_dates = pd.date_range(
        start=forecast_start,
        periods=forecast.samples.shape[1],
        freq='W'
    )
    
    # Calculate the median forecast
    forecast_median = forecast.quantile(0.5)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot past data
    plt.plot(past_data.index, past_data.values, label='Past Data', color='b', zorder=2)
    
    # Plot true future data if available
    if not future_data.empty:
        plt.plot(future_data.index, future_data.values, label='True Future', color='b', zorder=2)
    else:
        print(f"No true future data available for region {region}")
    
    # Plot forecast
    plt.plot(forecast_dates, forecast_median, color='g', label='Forecast Median', zorder=1)
    
    # Plot prediction intervals
    plt.fill_between(
        forecast_dates,
        forecast.quantile(0.1),
        forecast.quantile(0.9),
        color='g',
        alpha=0.3,
        label='80% Prediction Interval',
        zorder=1
    )
    
    # Set x-axis limits
    plt.xlim([past_data.index[0], forecast_dates[-1]])
    
    plt.title(f"Forecast for {region}")
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.tight_layout()
    filename = f"Plot_{region}.png"
    file_path = os.path.join(output_dir, filename)
    plt.savefig(file_path)
    plt.close()