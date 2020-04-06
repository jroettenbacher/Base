#!/usr/bin/env python
# script to average cloudnet input files to a certain time height resolution and select only relevant variables
# input cloudnet input netcdf files (from RV-Meteor)
# output averaged netcdf files

import sys
import os
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
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())
import numpy as np
import pandas as pd

# Load LARDA
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
# set end and start date
# gather command line arguments
method_name, args, kwargs = h._method_info_from_argv(sys.argv)

# gather argument
if 'date' in kwargs:
    date = str(kwargs['date'])
    begin_dt = dt.datetime.strptime(date + ' 00:00:00', '%Y%m%d %H:%M:%S')
    end_dt = dt.datetime.strptime(date + ' 00:00:00', '%Y%m%d %H:%M:%S')
else:
    begin_dt = dt.datetime(2020, 2, 1, 0, 0, 0)
    end_dt = dt.datetime(2020, 2, 19, 0, 0, 0)

dates = pd.date_range(begin_dt, end_dt).to_pydatetime()  # define all dates
outpath = "/projekt2/remsens/data/campaigns/eurec4a/LIMRAD94/30s_averages/"  # define path for output

# loop through all files in date range
for date in dates:
    t1 = time.time()
    # load data from larda
    Ze = larda.read("LIMRAD94_cn_input", "Ze", [date, (date + dt.timedelta(hours=23, minutes=59, seconds=30))],
                    [0, 'max'])
    # define new dimensions
    new_time = np.array([date + dt.timedelta(seconds=i) for i in range(0, int((24*60*60)), 30)])  # 30s timestep
    new_time = np.array([h.dt_to_ts(time) for time in new_time])  # convert to unix time stamps
    # 30m range steps, started from rounded down to the next 100 first range gate, and rounded up to the next 1000 last rg
    new_range = np.arange(Ze['rg'][0]//100*100, np.round(Ze['rg'][-1], -3)+30, 30)  # 300-15000 or 300-13020 m
    Ze = interpolate2d(Ze, new_time=new_time, new_range=new_range, method='linear')
    # turn mask from 1s and 0s to True and False
    Ze["mask"] = Ze["mask"] == 1
    # fill masked values with -999
    Ze["var"] = h.fill_with(Ze["var"], Ze["mask"], -999)
    # generate nc file
    container = {'Ze': Ze}  # create a container for the routine
    outfile = f"RV-METEOR_LIMRAD94_Ze_{date:%Y%m%d}.nc"
    if os.path.isfile(f"{outpath}{outfile}"):
        check1 = input(f"{outfile} already exists! Do you want to replace it? (This action cannot be undone!): ")
        if check1.startswith("y"):
            print(f"Deleting {outfile}...")
            os.remove(f"{outpath}{outfile}")
            print(f"Creating new nc file {outfile}")
            flag = nc.generate_30s_averaged_Ze_files(container, outpath)
            print(f"Generated nc file for {date} in {time.time() - t1}.")
        else:
            print("Moving on to next file.")
    else:
        print(f"Creating new nc file {outfile}")
        flag = nc.generate_30s_averaged_Ze_files(container, outpath)
        print(f"Generated nc file for {date} in {time.time() - t1}.")

