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
    '''Each entry contains a target and a start field. The start field '''
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

data_name = parameters["data_set"]
#Important Parameters
device = 'cpu'
SIZE = "large"
CTX = parameters["CTX"] # Prediction length. Number of points we are going to predict.
PDT = parameters["PDT"] # Context length. This is the length of the window from which we predict a single point.
BSZ = parameters["BSZ"] # Batch size; smaller sizes improve computation time 
PSZ = parameters['PSZ']
data_set = os.path.join("input", data_name )
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

# I just put this here to make organization easier when I'm playing around with the code.
# Creates a time stamped subfolder, which I place my predictions in later on
output_dir = f"Data_tests/Data: {data_name}, PDT:{PDT}, CTX: {CTX}, PSZ: {PSZ}, BSZ: {BSZ}, samples: {samples}"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save arrays to `.npy` files
np.save(f"Data_tests/forecast_PDT:{data_name}", forecasts[0].quantile(0.5))
np.save(f"Data_tests/truth:{data_name}", tss[0][CTX:])

print("Arrays saved successfully!")


for idx, region in enumerate(df.columns):

    print(f"Processing variable: {region}")
    ts = tss[0][idx] 
    forecast = forecasts[0] # We only have a single timeseries and forecast object so the lists have length 1
    
    # Extract the forecast for the specific variable. Forcast 
    forecast_median = forecast.quantile(0.5)[:, idx]  # Select only the median values for the current region

    # Create date range with the expected number of time steps
    forecast_dates = pd.period_range(
        start=forecast.start_date,
        periods=forecast.samples.shape[1],  # This should match prediction length PDT
        freq='W'
    ).to_timestamp()  # Convert forecast dates back to Timestamps for plotting (makes it compatible with matplotlib)
    
    # Ensure forecast_median has the correct length
    if forecast_median.shape[0] != forecast_dates.shape[0]:
        print(f"Mismatch in forecast length: forecast median has {forecast_median.shape[0]}, expected {forecast_dates.shape[0]}")
        continue  # Skip plotting if there's a mismatch
    
    # Plotting
    plt.figure(figsize=(10, 6))
    past_data = ts[-CTX:] # this throws out initial data but I don't think we care 
    plt.plot(past_data.index.to_timestamp(), past_data.values, label='Ground Truth', color='b')
    plt.plot(forecast_dates, forecast_median, color='g', label='Forecast Median')
    plt.fill_between(
        forecast_dates,
        forecast.quantile(0.1)[:, idx],  # 10th percentile for current region--lower bound
        forecast.quantile(0.9)[:, idx],  # 90th percentile for current region--upper bound
        color='g', alpha=0.3, label='80% Prediction Interval'
    )
    plt.title(f"Forecast for {region}")
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.tight_layout()
    filename = f"Plot_{region}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


#The MOIRAI paper uses this 
def mase(y_true, y_pred, seasonal_period=1):
    naive_forecast = y_true[:-seasonal_period]  # Naive forecast shifted by seasonal period
    denominator = np.mean(np.abs(y_true[seasonal_period:] - naive_forecast))
    numerator = np.mean(np.abs(y_true - y_pred))
    return numerator / denominator
def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100
def quantile_loss(y_true, y_pred, quantile):
    residual = y_true - y_pred
    return np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))
def coverage(y_true, lower_pred, upper_pred):
    within_bounds = (y_true >= lower_pred) & (y_true <= upper_pred)
    return np.mean(within_bounds) * 100


# Initialize storage for results
evaluation_results = []
results_raw = []

# Iterate over regions
for idx, region in enumerate(df.columns):
    print(f"Evaluating region: {region}")
    
    # Extract true and predicted values
    y_true = tss[0][idx].iloc[-PDT:].values  # True values for the last PDT points
    y_pred = forecasts[0].quantile(0.5)[:, idx]  # Median forecast
    y_lower = forecasts[0].quantile(0.1)[:, idx]  # 10th percentile forecast
    y_upper = forecasts[0].quantile(0.9)[:, idx]  # 90th percentile forecast

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mase_val = mase(y_true, y_pred, seasonal_period=1)  # Change seasonal_period if needed
    smape_val = smape(y_true, y_pred)
    ql = quantile_loss(y_true, y_pred, quantile=0.5)  # Evaluate quantile loss at 50% quantile
    cov80 = coverage(y_true, y_lower, y_upper)  # Coverage for 80% PI
    
    # Store results
    evaluation_results.append({
        "Region": region,
        "MAE": mae,
        "MASE": mase_val,
        "sMAPE": smape_val,
        "QL_50%": ql,
        "Coverage_80%": cov80
    })

# Convert to DataFrame for analysis
evaluation_df = pd.DataFrame(evaluation_results)
print(evaluation_df)


# Define a helper function to save plots
def save_plot(fig, filename, output_dir):
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)

# Plot MAE
fig = plt.figure(figsize=(10, 6))
plt.bar(evaluation_df["Region"], evaluation_df["MAE"], color='purple')
plt.title("Mean Absolute Error (MAE) by Region")
plt.xlabel("Region")
plt.ylabel("MAE")
plt.tight_layout()
save_plot(fig, "MAE_by_Region.png", output_dir)

# Plot MASE 
'''fig = plt.figure(figsize=(10, 6))
plt.bar(evaluation_df["Region"], evaluation_df["MASE"], color='blue')
plt.title("Mean Absolute Scaled Error (MASE) by Region")
plt.xlabel("Region")
plt.ylabel("MASE")
plt.axhline(1, color='red', linestyle='--', label="Naive Benchmark (MASE = 1)")
plt.legend()
plt.tight_layout()
save_plot(fig, "MASE_by_Region.png", output_dir)'''

# Plot sMAPE
fig = plt.figure(figsize=(10, 6))
plt.bar(evaluation_df["Region"], evaluation_df["sMAPE"], color='green')
plt.title("Symmetric Mean Absolute Percentage Error (sMAPE) by Region")
plt.xlabel("Region")
plt.ylabel("sMAPE (%)")
plt.tight_layout()
save_plot(fig, "sMAPE_by_Region.png", output_dir)

# Plot Coverage
fig = plt.figure(figsize=(10, 6))
plt.bar(evaluation_df["Region"], evaluation_df["Coverage_80%"], color='orange')
plt.title("Coverage (80%) by Region")
plt.xlabel("Region")
plt.ylabel("Coverage (%)")
plt.axhline(80, color='red', linestyle='--', label="Target Coverage (80%)")
plt.legend()
plt.tight_layout()
save_plot(fig, "Coverage_by_Region.png", output_dir)

# Plot Quantile Loss (QL)
'''fig = plt.figure(figsize=(10, 6))
plt.bar(evaluation_df["Region"], evaluation_df["QL_50%"], color='teal')
plt.title("Quantile Loss (QL 50%) by Region")
plt.xlabel("Region")
plt.ylabel("QL (50%)")
plt.tight_layout()
save_plot(fig, "QL_50_by_Region.png", output_dir)'''

print(f"Plots have been saved to: {output_dir}")