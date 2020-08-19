#!/usr/bin/env python
"""Script to analyze DSHIP data from RV Meteor
19.8.2020: SL found time gaps in DSHIP data and a possible time drift
"""

import time
import datetime as dt
import pandas as pd
import dask.dataframe as dd

path = "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP"
# # this code snippet checks if any column that should be a float64 column is one
# # and says if for one day this is not the case
# # somehow the missing data was set to -999-999.-999-999-999 on the 11.02.2020
# # data was redownloaded and the error was fixed that way, 19.08.2020, JR
# dates = pd.date_range(start=dt.datetime(2020, 1, 17), end=dt.datetime(2020, 2, 29))
# # Seapath attitude and heave data 1 or 10 Hz, choose file depending on date
# for date in dates:
#     if date < dt.datetime(2020, 1, 27):
#         file = f"{date:%Y%m%d}_DSHIP_seapath_1Hz.dat"
#     else:
#         file = f"{date:%Y%m%d}_DSHIP_seapath_10Hz.dat"
#     # set encoding and separator, skip the rows with the unit and type of measurement
#     seapath = pd.read_csv(f"{path}/{file}", encoding='windows-1252', sep="\t", skiprows=(1, 2))
#     # drop the datetime column
#     seapath = seapath.drop(columns='date time')
#     if all(seapath.dtypes == 'float64'):
#         print(f"True for {date}")
#     else:
#         print(f"False for {date}")
#

# read in all data and plot the difference between each time step
file = "*_DSHIP_seapath*"
seapath = dd.read_csv(f"{path}/{file}", encoding='windows-1252', sep="\t", skiprows=(1, 2),
                      parse_dates=['date time'])
# change colnames
seapath.columns = ['datetime', 'heading', 'heave', 'pitch', 'roll']


def diff(df, periods=1):
    before, after = (periods, 0) if periods > 0 else (0, -periods)
    return df.map_overlap(lambda df, periods=1: df.diff(periods),
                           periods, 0, periods=periods)


result = diff(seapath).compute()
