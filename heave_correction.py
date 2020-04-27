#!/bin/python

# Script to calculate heave rate from DSHIP data and plot it

########################################################################################################################
# library import
########################################################################################################################
import time
import datetime as dt
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

########################################################################################################################
# Data Read in
########################################################################################################################
start = time.time()
input_path = "/projekt2/remsens/data/campaigns/eurec4a/RV-METEOR_DSHIP"
plot_path = "/projekt1/remsens/work/jroettenbacher/plots/heave_correction"
# begin and end date
# TODO: get begin and end dt from LIMRAD94_to_Cloudnet_v2.py
date = dt.datetime(2020, 1, 17)
########################################################################################################################
# Seapath attitude and heave data 1 or 10 Hz
if date < dt.datetime(2020, 1, 27):
    file = f"{date:%Y%m%d}_DSHIP_seapath_1Hz.dat"
else:
    file = f"{date:%Y%m%d}_DSHIP_seapath_10Hz.dat"
rv_meteor = pd.read_csv(file, encoding='windows-1252', sep="\t", skiprows=(1, 2), index_col='date time')
rv_meteor.index = pd.to_datetime(rv_meteor.index, infer_datetime_format=True)
rv_meteor.index.name = 'datetime'
rv_meteor.columns = ['Heading [°]', 'Heave [m]', 'Pitch [°]', 'Roll [°]']
print(f"Done reading in Seapath data in {time.time() - start}")

########################################################################################################################
# Calculating Heave Rate
########################################################################################################################
t1 = time.time()
print("Calculating Heave Rate...")
heave_rate = np.ediff1d(rv_meteor["Heave [m]"]) / np.ediff1d(rv_meteor.index).astype('float64') * 1e9
heave_rate = pd.DataFrame({'Heave Rate [m/s]': heave_rate}, index=rv_meteor.index[1:])
rv_meteor = rv_meteor.join(heave_rate)
print(f"Done with heave rate calculation in {time.time() - t1}")

# calculate average heave rate for chirp time length 500 ms
rv_meteor_500ms = rv_meteor.resample('0.5S').mean()

########################################################################################################################
# Calculating Pitch induced heave
########################################################################################################################
