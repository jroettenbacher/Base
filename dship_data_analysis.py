#!/usr/bin/env python
"""Script to analyze DSHIP data from RV Meteor
19.08.2020: SL found time gaps in DSHIP data and a possible time drift
11.05.2021: Ludwig found some odd data on 12.2, 13.2, 15.2, 16.2
"""
# %%
import os
import sys
if os.getcwd().startswith("C:"):
    sys.path.append('C:/Users/Johannes/PycharmProjects/Base/larda')
else:
    sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import time
import datetime as dt
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
# %% functions


def read_dship_aeris(date, **kwargs):
    """Read in 1 Hz DSHIP data and return pandas DataFrame

    Args:
        date (str): yyyymmdd (eg. 20200210)
        **kwargs: kwargs for pd.read_csv (not all implemented) https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

    Returns: pd.DataFrame with 1 Hz DSHIP data

    """
    tstart = time.time()
    path = kwargs['path'] if 'path' in kwargs else "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP"
    skiprows = kwargs['skiprows'] if 'skiprows' in kwargs else (1, 2)
    nrows = kwargs['nrows'] if 'nrows' in kwargs else None
    cols = kwargs['cols'] if 'cols' in kwargs else None  # always keep the 0th column (datetime column)
    file = f"{path}/RV-METEOR_DSHIP_1Hz_{date}.dat"
    # set encoding and separator, skip the rows with the unit and type of measurement, set index column
    df = pd.read_csv(file, encoding='windows-1252', sep="\t", skiprows=skiprows, index_col='date time', nrows=nrows,
                     usecols=cols, na_values=' ')
    df.index = pd.to_datetime(df.index, format="%d/%m/%Y %H:%M:%S")
    df.index.rename('datetime', inplace=True)

    logger.info(f"Done reading in DSHIP data in {time.time() - tstart:.2f} seconds")

    return df


# %% set paths
if os.getcwd().startswith("C:"):
    # local
    path = "C:/Users/Johannes/Documents/EUREC4A/data/dship"
    plot_path = "C:/Users/Johannes/Documents/EUREC4A/data/dship"
else:
    # remote
    path = "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP"
    plot_path = "/projekt1/remsens/work/jroettenbacher/plots/dship"

# %%
# this code snippet checks if any column that should be a float64 column is one
# and says if for one day this is not the case
# somehow the missing data was set to -999-999.-999-999-999 on the 11.02.2020
# data was redownloaded and the error was fixed that way, 19.08.2020, JR
dates = pd.date_range(start=dt.datetime(2020, 1, 17), end=dt.datetime(2020, 2, 29))
# Seapath attitude and heave data 1 or 10 Hz, choose file depending on date
for date in dates:
    if date < dt.datetime(2020, 1, 27):
        file = f"{date:%Y%m%d}_DSHIP_seapath_1Hz.dat"
    else:
        file = f"{date:%Y%m%d}_DSHIP_seapath_10Hz.dat"
    # set encoding and separator, skip the rows with the unit and type of measurement
    seapath = pd.read_csv(f"{path}/{file}", encoding='windows-1252', sep="\t", skiprows=(1, 2))
    # drop the datetime column
    seapath = seapath.drop(columns='date time')
    if all(seapath.dtypes == 'float64'):
        print(f"True for {date}")
    else:
        print(f"False for {date}")

# %% read in all data and plot the difference between each time step
file = "*_DSHIP_seapath_1Hz.dat"
seapath = dd.read_csv(f"{path}/{file}", encoding='windows-1252', sep="\t", skiprows=(1, 2),
                      parse_dates=['date time'])
# change colnames
seapath.columns = ['datetime', 'heading', 'heave', 'pitch', 'roll']


def diff(df, periods=1):
    before, after = (periods, 0) if periods > 0 else (0, -periods)
    return df.map_overlap(lambda df, periods=1: df.diff(periods),
                           periods, 0, periods=periods)


result = diff(seapath.datetime).compute()
np.max(result)
np.min(result)
np.sum(np.isnan(result))
plt.plot(result[1:])
plt.savefig(f"{plot_path}/time_diff_seapath.png")
plt.close()

# %% test the read_dship_aeris function

date = '20200212'
dship = read_dship_aeris(date, path=path)
dship2 = read_dship_aeris('20200214', path=path)

dship_m = pd.concat([dship, dship2])

dship_m["SEAPATH.PSXN.Roll"].plot()
plt.show()
plt.close()
