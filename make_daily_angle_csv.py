#!/usr/bin/python
"""Script to read in inclination angles of LIMRAD94 and save them to daily csv files
input: IncEl and IncElA from larda
output: daily csv files
author: Johannes Roettenbacher"""

import sys
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import pyLARDA
import pyLARDA.helpers as h
import logging
import datetime as dt
import pandas as pd

# add logging
log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# where to save files
outpath = '/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/LIMRAD94/angles'
larda = pyLARDA.LARDA().connect('eurec4a')  # conect to campaign
# define date range
begin_dt = dt.datetime(2020, 1, 17, 0, 0, 0)
end_dt = dt.datetime(2020, 2, 28, 23, 59, 59)
# read in roll and pitch
radar_roll = larda.read("LIMRAD94_cn_input", "Inc_El", [begin_dt, end_dt])
radar_pitch = larda.read("LIMRAD94_cn_input", "Inc_ElA", [begin_dt, end_dt])
# extract unix time stamp and convert it to datetime
unix_time = radar_roll['ts']
datetime = [h.ts_to_dt(t) for t in unix_time]
# create data frame for output
df = pd.DataFrame({'datetime': datetime, 'unix_time': unix_time,
                   'roll': radar_roll['var'], 'pitch': radar_pitch['var']})

# get unique dates in data frame and loop through the to make daily files
for date in df.datetime.dt.date.unique():
    filename = f"{outpath}/RV-Meteor_cloudradar_attitude-angles_{date:%Y%m%d}.csv"
    tmp_df = df.loc[df.datetime.dt.date == date]  # select only rows which match the current date
    tmp_df.to_csv(filename, index=False)  # save to csv
