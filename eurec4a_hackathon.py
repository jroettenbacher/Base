#!/usr/bin/env python
"""Hackathon stuff 07.09.2021
author: Johannes Röttenbacher
"""

# %%
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gsw import distance

def read_dship(date, **kwargs):
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
    file = f"{path}/RV-Meteor_DSHIP_all_1Hz_{date}.dat"
    # set encoding and separator, skip the rows with the unit and type of measurement, set index column
    df = pd.read_csv(file, encoding='windows-1252', sep="\t", skiprows=skiprows, index_col='date time', nrows=nrows,
                     usecols=cols, na_values='-999.0')
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
    df.index.rename('datetime', inplace=True)

    print(f"Done reading in DSHIP data in {time.time() - tstart:.2f} seconds")

    return df


def convert_to_decimal_degree(deg_min: str):
    split_input = deg_min.split(" ")
    deg = float(split_input[0][:-1])
    dez = float(split_input[1][:-1])/60
    dez_deg = deg + dez

    return dez_deg


# %% set up paths
date_range = pd.date_range("20200117", "20200219")
dates_seg1 = ["20200120"]
dship = pd.concat([read_dship(date) for date in dates_seg1])
var = "SEAPATH.PSXN.Heading"

# %% get CTD start time
events = "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP/20200117-20200301_RV-Meteor_events.dat"
events_df = pd.read_csv(events, sep="\t", encoding="cp1252")
ctd_starts = events_df[events_df.Action == "in the water"]
ctd_start_times = ctd_starts.loc[:, ["date time", "Latitude", "Longitude"]]
ctd_start_times["Latitude"] = [convert_to_decimal_degree(lat) for lat in ctd_start_times["Latitude"]]
ctd_start_times["Longitude"] = [convert_to_decimal_degree(lon) for lon in ctd_start_times["Longitude"]]

# %% get 1 km radius around CTD postion
lat = dship["SYS.STR.PosLat"].values
lon = dship["SYS.STR.PosLon"].values

distances = distance(lon, lat)
distances = np.append(distances, np.nan)
dship = dship.assign(distance=distances)

# %% calculate cumulative distance


# %% get time of first point inside the circle before the CTD
# get time of last point inside the circle after the CTD
# set heading during CTD to Null
# filter by heading and get first and last time step before change of heading

start_1 = "2020-01-20 18:00"
end_1 = "2020-01-21 08:36"

# %%
fig, ax = plt.subplots()
lat_lon = ax.scatter(x="SYS.STR.PosLon", y="SYS.STR.PosLat", data=dship, c=dship.index.values.astype(float))
dship[start_1].plot.scatter(x="SYS.STR.PosLon", y="SYS.STR.PosLat", ax=ax, label="start_1", c="r")
dship[end_1].plot.scatter(x="SYS.STR.PosLon", y="SYS.STR.PosLat", ax=ax, label="end_1", c="g")
plt.colorbar(lat_lon)
ax.grid()
ax.legend()
plt.show()
plt.close()