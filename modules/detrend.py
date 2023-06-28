#!/usr/bin/env python3

"""
This module contains the functionality needed to detrend time series of crop yields.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, TheilSenRegressor, HuberRegressor, RANSACRegressor
import statsmodels.api as sm
from collections.abc import Iterable
from sklearn.preprocessing import StandardScaler

def plot_time_series(time_series, to_plot=["value"], labels=["raw yield"], ax=None):
    """
    plot a time series of yield data including a trend estimation.
    params:
        time_series: dataframe, raw data for the plot and the trend estimation
        to_plot: string, column name that indicates which column should be plotted
        trend_estimator: string, indicating the estimation technique
        ax: pyplot ax to plot the data on
    returns:
        plot: pyplot plot
    """
    data = time_series.copy()
    data = data.dropna(subset=to_plot)
    
    column_to_label = dict(zip(to_plot, labels))
    
    if ax is None:
        ax = plt.gca()
    for col in to_plot:
        plot = ax.plot(data["year"], data[col], label=column_to_label[col]);
    
   
    #    plot = ax.plot(data["year"], m*data["year"] + b);
            
    ax.set_ylabel(str(data["quantity"].unique()[0].lower()));
    ax.legend();
    ax.set_xlabel("year");
    
    return plot
    
    
def regression_detrending(time_series, decomposition="additive", polynomial=1, regressor=LinearRegression()):
    """
    apply regression detrending.
    params:
        time_series: dataframe to be detrended
        decomposition: string, one of additive or multiplicative
        polynomial: int, the order of regression
        regressor: regressor object from sklearn.linear_model
    returns:
        data: dataframe containing new column with detrended series
    """
    data = time_series.copy()
    
    x = np.array(data["year"]).reshape(-1, 1)
    y = np.array(data["value"]).reshape(-1, 1)
    
    # create array of the specified order of polynomials
    result = np.zeros((x.shape[0], polynomial+1))
    result[:, 0] = x[:, 0] 
    for i in range(2, polynomial+1):
        result[:, i-1] = np.power(x, i)[:, 0]
    x = result
    
    reg = regressor.fit(x, np.ravel(y))
    y_new = reg.predict(x)
   
    y_new = list(flatten(y_new))
    y = list(flatten(y))
    
    data["trend_estimated [t/ha]"] = y_new 
   
    if decomposition not in ["additive", "multiplicative"]: 
        print("decomposition must be additive or multiplicative")
        return None
    if decomposition == "additive":
        data["value_detrended [t/ha]"] = [a_i - b_i for a_i, b_i in zip(y, y_new)]
    if decomposition == "multiplicative":
        data["value_detrended [%]"] = [(a_i - b_i)/b_i for a_i, b_i in zip(y, y_new)]
    
    return data

def moving_average_detrending(time_series, decomposition="additive", window=3):
    """
    apply moving average detrending.
    params:
        time_series: dataframe to be detrended
        decomposition: string, one of additive or multiplicative
        window: int, hyperparam to be passed rolling, defines window size
    returns:
        data: dataframe containing new column with detrended series
    """
    data = time_series.copy()
    
    rolling_mean = data["value"].rolling(window=window, closed="left").mean()
    
    data["trend_estimated [t/ha]"] = rolling_mean 
    
    if decomposition not in ["additive", "multiplicative"]: 
        print("decomposition must be additive or multiplicative")
        return None
    if decomposition == "additive":
        data["value_detrended [t/ha]"] = data["value"] - rolling_mean
    if decomposition == "multiplicative":
        data["value_detrended [%]"] = (data["value"] - rolling_mean)/rolling_mean
    
    return data

def iizumi_detrending(time_series, decomposition="multiplicative", window=3):
    """
    apply moving average detrending.
    params:
        time_series: dataframe to be detrended
        decomposition: string, one of additive or multiplicative
        window: int, hyperparam to be passed rolling, defines window size
    returns:
        data: dataframe containing new column with detrended series
    """
    data = time_series.copy()
    
    rolling_mean = data["value"].rolling(window=window, closed="left").mean()
    
    data["trend_estimated [t/ha]"] = rolling_mean 
    
    if decomposition not in ["additive", "multiplicative"]: 
        print("decomposition must be additive or multiplicative")
        return None
    if decomposition == "additive":
        print("Iizumi does not implement an additive decomposition. Choose multiplicative.")
    if decomposition == "multiplicative":
        data["value_detrended [%]"] = data["value"].diff()/rolling_mean
    
    return data


def lowess_detrending(time_series, decomposition="additive", frac=.9):
    """
    apply lowess detrending to time series.
    params:
        time_series: dataframe to be detrended
        decomposition: string, one of additive or multiplicative
        frac: float, hyperparam to be passed to lowess
    returns:
        data: dataframe containing new column with detrended series
    """
    data = time_series.copy()
    
    x = data["year"]
    y = data["value"]
    lowess = sm.nonparametric.lowess(y, x, frac=frac)
    lowess_y = list(zip(*lowess))[1]
    
    data["trend_estimated [t/ha]"] = lowess_y
    
    if decomposition not in ["additive", "multiplicative"]: 
        print("decomposition must be additive or multiplicative")
        return None
    if decomposition == "additive":
        y_detrended = y - lowess_y
        data["value_detrended [t/ha]"] = y_detrended
    if decomposition == "multiplicative":
        y_detrended = (y-lowess_y)/lowess_y
        data["value_detrended [%]"] = y_detrended
        
    return data

def standardize(time_series, to_standardize="value"):
    
    data = time_series.copy()
    data[to_standardize + "_standardized"] = StandardScaler().fit_transform(np.array(data[to_standardize]).reshape(-1, 1))
    
    return data

def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x