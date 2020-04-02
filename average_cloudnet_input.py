#!/usr/bin/env python
# script to average cloudnet input files to a certain time height resolution and select only relevant variables
# input cloudnet input netcdf files (from RV-Meteor)
# output averaged netcdf files

import sys
import time
import datetime as dt
sys.path.append(r'C:\Users\Johannes\PycharmProjects\Base\larda')
sys.path.append('.')
import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.NcWrite as nc
import logging
import numpy as np
import pandas as pd
import functions_jr as jr

# Load LARDA
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
# set end and start date
begin_dt = dt.datetime(2020, 1, 17, 0, 0, 0)
end_dt = dt.datetime(2020, 2, 20, 0, 0, 0)
dates = pd.date_range(begin_dt, end_dt)  # define all dates
path = "./data/cloudnet_input/"  # define path to cloudnet files

# loop through all files in date range
# for date in dates:
# load data from larda
date = begin_dt
Ze = larda.read("LIMRAD94_cn_input", "Ze", [date, (date + dt.timedelta(days=1))], [0, max])
# define new dimensions
new_time = np.array([date + dt.timedelta(seconds=i) for i in range(0, int((24*60*60)), 30)])  # 30s timestep
# new_range =
file = f"{path}{date:%Y%m%d}_000000-240000_LIMRAD94.nc"
