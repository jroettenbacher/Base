#!/usr/bin/env python
# script to average cloudnet input files to a certain time height resolution and select only relevant variables
# input cloudnet input netcdf files (from RV-Meteor)
# output averaged netcdf files

import sys
import time
import datetime as dt
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import pyLARDA
import pyLARDA.helpers as h
from pyLARDA.Transformations import interpolate2d
import pyLARDA.NcWrite as nc
import logging
log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())
import numpy as np
import pandas as pd

# Load LARDA
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
# set end and start date
begin_dt = dt.datetime(2020, 1, 17, 0, 0, 0)
end_dt = dt.datetime(2020, 2, 20, 0, 0, 0)
dates = pd.date_range(begin_dt, end_dt).to_pydatetime()  # define all dates
outpath = "/projekt2/remsens/data/campaigns/eurec4a/LIMRAD94/cloudnet_input/"  # define path for output

# loop through all files in date range
# for date in dates:
t1 = time.time()
# load data from larda
date = dates[0]
Ze = larda.read("LIMRAD94_cn_input", "Ze", [date, (date + dt.timedelta(days=1))], [0, 'max'])
# define new dimensions
new_time = np.array([date + dt.timedelta(seconds=i) for i in range(0, int((24*60*60)), 30)])  # 30s timestep
new_time = np.array([h.dt_to_ts(time) for time in new_time])  # convert to unix time stamps
# 30m range steps, started from rounded down to the next 100 first range gate, and rounded up to the next 1000 last rg
new_range = np.arange(Ze['rg'][0]//100*100, np.round(Ze['rg'][-1], -3)+30, 30)  # 300-15000 or 300-13020 m
Ze = interpolate2d(Ze, new_time=new_time, new_range=new_range, method='linear')
# generate nc file
container = {'Ze': Ze}  # create a container for the routine
flag = nc.generate_30s_averaged_Ze_files(container, outpath)
print(f"Generated nc file for {date} in {time.time() - t1}.")

