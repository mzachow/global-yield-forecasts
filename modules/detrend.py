#!/usr/bin/env python3

"""
This module contains the functionality needed to detrend time series of crop yields.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def _plot_linear_trend(data, to_plot="value", figsize=(8,5)):
    ax = data.plot("year", to_plot, legend=False, figsize=figsize);
    if to_plot == "value":
        m, b = np.polyfit(data["year"], data[to_plot], 1);
        plt.plot(data["year"], m*data["year"] + b);
    else:
        plt.plot(data["year"], data.shape[0]*[data[to_plot].mean()])
    ax.set_ylabel(str(data["quantity"].unique()[0].lower() + " [" + str(data["unit"].unique()[0] + "]")));
    ax.set_xlabel("year");
    
def _linear_detrending(data):
    df = data[["year", "value"]].copy()
    li = []
    for year in df["year"].unique():
        yields_to_calculate_trend = df.loc[df["year"] != year].copy().reset_index(drop=True)
        yield_to_be_adjusted = df.loc[df["year"] == year].copy().reset_index(drop=True)
        reg = LinearRegression()
        slope_cv = reg.fit(yields_to_calculate_trend["year"].values.reshape(-1,1), yields_to_calculate_trend["value"]).coef_[0]

        yield_to_be_adjusted["value_detrended"] = yield_to_be_adjusted["value"] + (slope_cv * (df["year"].max() - yield_to_be_adjusted["year"].astype(int)))
        li.append(yield_to_be_adjusted.round(2))
        
    df_cv = (pd
            .concat(li, axis=0, ignore_index=False)
            .reset_index(drop=True)
            .apply(pd.to_numeric)
            .assign(quantity=data["quantity"].unique()[0], unit=data["unit"].unique()[0]))
    
    return df_cv

def _moving_average_anomalies(data):
    data["value_anomaly"] = data["value"].diff()/data["value"].rolling(window=3, closed="left").mean()
    return data

def _standardize(data):
    
    return data
