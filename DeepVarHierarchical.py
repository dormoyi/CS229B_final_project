import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# local_gluonts_path = "/home/nyss/disque_tera/1_sf/2 cs299b/CS229B_final_project/gluonts"

# # Add the local path to the Python path
# sys.path.insert(0, os.path.abspath(local_gluonts_path))

USE_INT64_TENSOR_SIZE=1

from gluonts.mx.model.deepvar_hierarchical import DeepVARHierarchicalEstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.hierarchical import HierarchicalTimeSeries
from gluonts.model.predictor import Predictor
from pathlib import Path

import gc
gc.collect()


sns.set_theme(style="darkgrid")

load_df = pd.read_csv('data/load.csv')
hierarchy_df = pd.read_csv("data/hierarchy.csv")
humidity_df = pd.read_csv("data/relative humidity.csv")
temperature_df = pd.read_csv("data/temperature.csv")

load_df = pd.melt(load_df, id_vars=["meter_id", "date"], value_vars=load_df.columns.difference(["meter_id", "date"]),
                                var_name="hour", value_name="load")
load_df["hour"] = load_df["hour"].str.strip("h").astype(int) - 1
load_df["timestamp"] = pd.to_datetime(load_df["date"] + " " + load_df["hour"].astype(str) + ":00:00", format="%m/%d/%Y %H:%M:%S")
load_df["meter_id"] = load_df["meter_id"].astype(int)
load_df = load_df.drop(columns=["date", "hour"])

df_pivoted = load_df.pivot(index='timestamp', columns='meter_id', values='load')

nan_values = df_pivoted.isna().sum()

# show the columns with more than 10% of nan values
nan_values[nan_values > 0.1 * len(df_pivoted)]

# drop columns with more than 10% of nan values
df_pivoted = df_pivoted.drop(columns=nan_values[nan_values > 0.1 * len(df_pivoted)].index)

# drop the 236 column because it has nan values in the test set
df_pivoted = df_pivoted.drop(columns=[236])

# also drop 13 and 144 before of 0 ans nans
df_pivoted = df_pivoted.drop(columns=[28, 453])

ts_mapping = {}
num_bottom_ts = len(df_pivoted.columns)
S = []
counter = 0

# add top time-seroes
name = "top"
ts_mapping[name] = counter
counter += 1
S.append([1 for i in range(num_bottom_ts)])

# add aggregated time-series
for i, row in hierarchy_df.iterrows():
    meter_id, mid_level, aggregate = row
    name = aggregate
    if aggregate not in ts_mapping and meter_id in df_pivoted.columns:
        ts_mapping[name] = counter
        counter += 1
        hierarchy = [0 for i in range(num_bottom_ts)]
        hierarchy[df_pivoted.columns.get_loc(meter_id)] = 1
        S.append(hierarchy)
    elif aggregate in ts_mapping and meter_id in df_pivoted.columns:
        hierarchy = S[ts_mapping[aggregate]]
        hierarchy[df_pivoted.columns.get_loc(meter_id)] = 1
        S[ts_mapping[aggregate]] = hierarchy


# add mid_level time-series
for i, row in hierarchy_df.iterrows():
    meter_id, mid_level, aggregate = row
    name = mid_level
    if mid_level not in ts_mapping and meter_id in df_pivoted.columns:
        ts_mapping[name] = counter
        counter += 1
        hierarchy = [0 for i in range(num_bottom_ts)]
        hierarchy[df_pivoted.columns.get_loc(meter_id)] = 1
        S.append(hierarchy)
    elif mid_level in ts_mapping and meter_id in df_pivoted.columns:
        hierarchy = S[ts_mapping[mid_level]]
        hierarchy[df_pivoted.columns.get_loc(meter_id)] = 1
        S[ts_mapping[mid_level]] = hierarchy

# add bottom time-series
for i in range(num_bottom_ts):
    name = df_pivoted.columns[i]
    if name not in ts_mapping:
        ts_mapping[name] = counter
        counter += 1
        hierarchy = [0 for i in range(num_bottom_ts)]
        hierarchy[i] = 1
        S.append(hierarchy)

S = np.array(S)

assert (S[1] + S[2] == S[0]).all()
assert (S[3:18].sum(axis=0) == S[0]).all()


df_pivoted.columns = range(1, len(df_pivoted.columns) + 1)

# Make sure the dataframe has `PeriodIndex` by explicitly casting it to `PeriodIndex`.
df_pivoted2 = df_pivoted.to_period()

# replace nan valyes with closest non-nan value
df_pivoted2 = df_pivoted2.fillna(method='backfill')

assert S.shape[0] == len(ts_mapping)
assert S.shape[1] == len(df_pivoted2.columns)


humidity_df = pd.read_csv("data/relative humidity.csv")
temperature_df = pd.read_csv("data/temperature.csv")
humidity_df["timestamp"] = pd.to_datetime(humidity_df["date"] + " " + (humidity_df["hr"] - 1).astype(str) + ":00:00", format="%d%b%Y %H:%M:%S")
temperature_df["timestamp"] = pd.to_datetime(temperature_df["date"] + " " + (temperature_df["hr"] - 1).astype(str) + ":00:00", format="%d%b%Y %H:%M:%S")
humidity_df = humidity_df.drop(columns=["date", "hr"])
temperature_df = temperature_df.drop(columns=["date", "hr"])
features_df = temperature_df.merge(humidity_df, on="timestamp", how="outer")
print(len(features_df))
# set timestamp as index
features_df = features_df.set_index("timestamp").to_period("H")

df_pivoted2_trunc = df_pivoted2.iloc[-2000:]
features_df_trunc = features_df.iloc[-2000:]

hts = HierarchicalTimeSeries(
    ts_at_bottom_level=df_pivoted2_trunc,
    S=S,
)

dataset = hts.to_dataset()
# 4096000000 vs. 2147483647 - 25000, 2000
# 4096000000 vs. 2147483647 - 50000, 2000
# 4,096,000,000 vs. 2147483647 - 10000, 2000
# 4,096,000,000 vs. 2147483647 - 15999, 2000
# 8192000000 vs. 2,147,483,647 - 25000, 4000
prediction_length = 24
hts_train = HierarchicalTimeSeries(
    ts_at_bottom_level=df_pivoted2_trunc.iloc[:-prediction_length, :],
    S=S,
)
hts_test_label = HierarchicalTimeSeries(
    ts_at_bottom_level=df_pivoted2_trunc.iloc[-prediction_length:, :],
    S=S,
)

features_df_trunc = features_df_trunc.iloc[:-prediction_length, :]

dataset_train = hts_train.to_dataset(feat_dynamic_real=features_df_trunc)
# dataset_test_label = hts_test_label.to_dataset()

predictor_input = hts_train.to_dataset(feat_dynamic_real=features_df_trunc)
# predictor_input = hts_test_label.to_dataset()


estimator = DeepVARHierarchicalEstimator(to_period
    freq=hts_train.freq,
    prediction_length=prediction_length,
    trainer=Trainer(epochs=1),
    S=S,
)

experiment = "output/testDynamic"

def train(dataset):
    print("Training")

    
    directory_path = Path(experiment)
    directory_path.mkdir(parents=True, exist_ok=True)


    predictor = estimator.train(dataset)

    print('Saving the model')
    predictor.serialize(Path(experiment))

train(dataset_train)

print('Loading the model: ', experiment)
predictor = Predictor.deserialize(Path(experiment))

print('Predicting')
# forecast_it = predictor.predict(dataset)

# There is only one element in `forecast_it` containing forecasts for all the time series in the hierarchy.
# forecasts = next(forecast_it)


forecast_it = predictor.predict(predictor_input)

from gluonts.evaluation import MultivariateEvaluator

evaluator = MultivariateEvaluator()
agg_metrics, item_metrics = evaluator(
    ts_iterator=[hts_test_label.ts_at_all_levels],
    fcst_iterator=forecast_it,
)

print(
    f"Mean (weighted) quantile loss over all time series: "
    f"{agg_metrics['mean_wQuantileLoss']}"
)

print(
    f"Mean (weighted) absolute error over the top time series: "
    f"{agg_metrics['0_abs_error']/prediction_length}"
)   

# check MAE with missing values


print(
    f"Mean (weighted) squared error over the top time series: "
    f"{agg_metrics['0_RMSE']}"
)

# then look at table to determine each one of the metrics 
# print(agg_metrics.keys())

# Plot the forecasts for the top time series in the hierarchy

forecast_it = predictor.predict(predictor_input)

forecasts = next(forecast_it)


point_estimations = np.mean(forecasts.samples, axis=0)
assert point_estimations.shape == (prediction_length, 160)
top_series = point_estimations[ :, 0]
assert top_series.shape == (prediction_length,)

top_series_label = np.array(hts_test_label.ts_at_all_levels[0][-prediction_length:])

# plt.figure(figsize=(12, 5))
# plt.plot(top_series_label, label="target")
# plt.plot(top_series, label="forecast")
# plt.legend()
# plt.grid(which="both")
# plt.show()


# use cs224n environment

# add additional variables

# get colab pro and run the whole model on it

# check hyperparameters (and do some grid search?)
# get loss and training curves 

# get plots and add them to the report

# evaluate reconciliation level

# check what time-series to take out for benchmarking
# add inference time in benchmarking (and memory usage)
# QUESTIONS how is train test valid made especially inside the training (when is loss computed?)

# get number of parameters in the model?

