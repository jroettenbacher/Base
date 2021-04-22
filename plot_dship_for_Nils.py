#!/usr/bin/env python
"""Plot RV Meteor DSHIP data
author: Johannes Roettenbacher
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def read_dship(date, **kwargs):
    """Read in 1 Hz DSHIP data and return pandas DataFrame

    Args:
        date (str): yyyymmdd (eg. 20200210)
        **kwargs: kwargs for pd.read_csv (not all implemented) https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

    Returns: pd.DataFrame with 1 Hz DSHIP data

    """
    tstart = time.time()
    # check for keyword arguments else use default values
    path = kwargs['path'] if 'path' in kwargs else "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP/upload_to_aeris"
    skiprows = kwargs['skiprows'] if 'skiprows' in kwargs else (1, 2)
    nrows = kwargs['nrows'] if 'nrows' in kwargs else None
    # when defining cols, always keep the 0th column (datetime column)!
    cols = kwargs['cols'] if 'cols' in kwargs else None
    file = f"{path}/RV-METEOR_DSHIP_1Hz_{date}.dat"
    # set encoding and separator, skip the rows with the unit and type of measurement, set index column
    df = pd.read_csv(file, encoding='windows-1252', sep="\t", skiprows=skiprows, index_col='date time', nrows=nrows,
                     usecols=cols)
    df.index = pd.to_datetime(df.index, format="%d/%m/%Y %H:%M:%S")
    df.index.rename('datetime', inplace=True)
    # TODO: rename columns for easier access by label

    print(f"Done reading in DSHIP data in {time.time() - tstart:.2f} seconds")

    return df


if __name__ == "__main__":
    # local path
    indir = "C:/Users/Johannes/Documents/bachelortheses/2021_NilsWalper"
    # remote path
    # indir = "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP/upload_to_aeris"
    date = "20200214"
    df = read_dship(date, path=indir)

    # plot time series of a variable, for quicklooks pandas is okay, otherwise use pyplot
    # the meteorological parameters are only available every minute but the data has a resolution of 1 Hz!
    # TODO: drop Nan values either just before plotting or for the whole dataset
    df_new = df.dropna()
    # pandas internal plotting (nice time formatting)
    df["WEATHER.PBWWI.HumidityPort"].dropna().plot()
    plt.show()
    plt.close()

    # pyplot plotting, needs more attention to detail
    fig, ax = plt.subplots()  # one way of plotting with matplotlib, allows for very fine tuning
    ax.plot(df["WEATHER.PBWWI.HumidityPort"].dropna())
    plt.show()
    plt.close()

    # TODO: plot other meteorological parameters
    # TODO: add titles and labels
    # TODO: optional combine parameters in one figure by adding one panel per parameter

    # make absolute histogram of humidity
    # plt.hist() uses np.histogram() and returns a tuple with the counts and the bin_edges
    hist = plt.hist(df_new["WEATHER.PBWWI.HumidityPort"])  # the other way of plotting with matplotlib, easier to understand
    plt.show()
    plt.close()

    # TODO: plot the relative frequency of occurence (tip: read the documentation of np.histogram())
    # TODO: plot histograms of other parameters
    # TODO: add labels

    # combine temperature and humidity in a scatter plot
    rel_hum = df_new["WEATHER.PBWWI.HumidityPort"]
    temp = df_new["WEATHER.PBWWI.AirTempPort"]
    plt.scatter(rel_hum, temp)
    plt.show()
    plt.close()

    # TODO: add labels
    # TODO: combine different parameters

    # add a regression line
    fit_coeffs = np.polyfit(rel_hum, temp, deg=1)  # returns intercept and slope in an array
    m, b = fit_coeffs[0], fit_coeffs[1]
    plt.scatter(rel_hum, temp)
    plt.plot(rel_hum, m * rel_hum + b, color="red")
    plt.show()
    plt.close()

    # TODO: add labels
    # TODO: try to add the slope and intercept as text to the plot
    # TODO: optional, calculate the R² values (coefficient of fit)

    # there is a statistical plotting module for python called seaborn which was designed for plots like this
    # if you want you can dive more into that, maybe it saves you some time now and then
    # I'm not too deep into it though. I think it is heavily inspired by R
    import seaborn as sns
    # create scatterplot with regression line
    sns.regplot(x=rel_hum, y=temp, ci=None)  # ci=None removes confidence intervals
    plt.show()
    plt.close()

    # add a cloud parameter: Liquid Water Path (LWP)
    # netCDF files can be easily read in and plotted with xarray, which also reads in all the available metadata
    # it works a lot like pandas and saves variables as numpy arrays under the hood
    file = "sups_met_mwr00_l2_clwvi_v00_20200214000000.nc"
    ds = xr.open_dataset(f"{indir}/{file}")
    ds["clwvi"].plot()
    plt.show()
    plt.close()

    # you can also access them via the dot notation, .values returns the numpy array
    ds.clwvi #.values

    # TODO: resample the LWP to minutely data (http://xarray.pydata.org/en/stable/generated/xarray.Dataset.resample.html)
    # TODO: plot LWP together with base meteorological parameters

