import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

local_gluonts_path = "/home/nyss/disque_tera/1_sf/2 cs299b/CS229B_final_project/gluonts"

# Add the local path to the Python path
sys.path.insert(0, os.path.abspath(local_gluonts_path))

# import my_gluonts.mx.model.deepvar_hierarchical
from gluonts.mx.model.deepvar_hierarchical import DeepVARHierarchicalEstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.hierarchical import HierarchicalTimeSeries


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

df_pivoted2_trunc = df_pivoted2.iloc[-10000:]

hts = HierarchicalTimeSeries(
    ts_at_bottom_level=df_pivoted2_trunc,
    S=S,
)

dataset = hts.to_dataset()

print("training")

prediction_length = 24
estimator = DeepVARHierarchicalEstimator(
    freq=hts.freq,
    prediction_length=prediction_length,
    trainer=Trainer(epochs=2),
    S=S,
)

# save the model and predict, add variables when training


predictor = estimator.train(dataset)

forecast_it = predictor.predict(dataset)

# There is only one element in `forecast_it` containing forecasts for all the time series in the hierarchy.
forecasts = next(forecast_it)
