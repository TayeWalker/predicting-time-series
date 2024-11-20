import torch
import matplotlib
matplotlib.use('Agg') #agg is a backend for non-interactive enviroments
import matplotlib.pyplot as plt
import pandas as pd


from gluonts.evaluation.backtest import make_evaluation_predictions #function to generate forecasts
from gluonts.dataset.field_names import FieldName #constants representing field names in data entries.
from gluonts.model.forecast import SampleForecast #forecast object containing samples of future values.

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule #model

import os
from datetime import datetime

# Custom dataset class for handling the multivariate data. It seems clunky to use an iterator becuase we are only yielding one value, but I believe that's the expected form.
class CustomTimeSeriesDataset:
    '''Each entry contains a target and a start field. The start field '''
    def __init__(self, df, prediction_length, freq):
        self.df = df
        self.prediction_length = prediction_length
        self.freq = freq
    
    def __iter__(self):
        yield {
            FieldName.START: self.df.index[0],  # Gives the start date
            FieldName.TARGET: self.df.values.T,  # 2D shape: (variables, time steps). This field contains almost all the information
            FieldName.ITEM_ID: "multivariate_series" #
        }
    def __len__(self):
        return 1 #We are only yielding one value so this is true

# I just put this here to make organization easier when I'm playing around with the code.
# Creates a time stamped subfolder, which I place my predictions in later on
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"PDT = 40, CTX = 180"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#Important Parameters
device = 'cpu'
SIZE = "large"
PDT = 40  # Prediction length
CTX = 180  # Context length
PSZ = "auto"
BSZ = 32 # batch size; smaller sizes improve computation time 
samples = 100 #Times that we sample from the distribution

# Load Data
df = pd.read_csv("unofficial_data.csv", parse_dates=['Date'])
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
    past_data = ts[-CTX:]
    plt.plot(past_data.index.to_timestamp(), past_data.values, label='Past Data', color='b')
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
    plt.ylabel('Value')
    plt.tight_layout()
    filename = f"Plot_{region}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
