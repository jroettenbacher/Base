#!/usr/bin/python

"""Script to correct the time shift in ceilometer data from the RV-Meteor during Eurec4a
The ceilometer was not synced to ship time but was lagging ~ 6 minutes behind.
On Jan 25 2020 at 18:47 UTC the ceilometer PC was synced to ship time.
The time in the nc files comes from the ceilometer itself, which gets synced with the ceilometer PC from the ceilometer
PC. Thus the time correction was not immediately applied but it took some time for the PC to sync its time onto the
ceilometer.
A close to 6 minute time skip can be observed from Jan 26 4:56:46 UTC to 5:02:44 UTC. Since the ceilometer measures
about every 10 seconds, the real time difference was:
5:48 minutes or 348 seconds
Each measurement before the time skip can thus be adjusted by that.
input: ceilometer nc files
output: ceilometer nc files time corrected
"""

import os
import re
import datetime as dt
import pandas as pd
import xarray as xr
import numpy as np

inpath = "/projekt2/remsens/data/campaigns/eurec4a/RV-METEOR_CEILOMETER/CHM170158/2020/01"
outpath = "/projekt2/remsens/data/campaigns/eurec4a/RV-METEOR_CEILOMETER/time_corrected"
# date range for which time needs to be shifted
begin_dt = dt.datetime(2020, 1, 16, 0, 0, 0)
end_dt = dt.datetime(2020, 1, 26, 0, 0, 0)
date_range = pd.date_range(begin_dt, end_dt)

all_files = sorted([f for f in os.listdir(inpath) if f.endswith(".nc")])
infiles = []
for file in all_files:
    # extract date from filename
    pattern = r"(?P<date>\d{8})_FSMETEOR_CHM170158.nc"
    m = re.search(pattern, file)
    date_from_file = dt.datetime.strptime(m.group('date'), "%Y%m%d")
    # list only file names which fall within date range to be time shifted
    if date_from_file in date_range:
        infiles.append(file)

os.chdir(inpath)
# read in all nc files in date range
ds = xr.open_mfdataset(infiles, parallel=True, combine='nested', concat_dim='time', engine='netcdf4',
                       decode_times=True)
# search for maximum difference between consecutive time steps
ix = np.asarray(np.where(ds.time.diff('time') == ds.time.diff('time').max())).flatten()
time_res = np.asarray(ds.time.diff('time').median())  # median time resolution of measurement
# retrieve indices before and after biggest time skip
indices = sorted(np.concatenate((ix - 1, ix, ix + 1)))
time_sub = ds.time[indices].values.copy()  # extract times at time skip
correction = time_sub[2] - time_sub[1] - time_res  # calculate time shift for correction
time = ds.time.values.copy()  # extract time variable from data set
time[0:indices[2]] = time[0:indices[2]] + correction  # move time skip to beginning of Jan 16

# correct dataset, add new attributes
ds.assign_coords(time=time)
ds.time.assign_attrs({'long_name': "time UTC",
                      'axis': "T"})
ds["time"].encoding = {'units': "seconds since 1904-01-01 00:00:00.000 00:00", 'calendar': "standard",
                       'dtype': 'i4'}
ds = ds.assign_attrs(comment="This file was corrected for a time lag. It was lagging behind 348 seconds. "
                             "That error was corrected on Jan 26 04:56:46 UTC, which introduced a time skip. "
                             "This time skip was moved to the beginning of Jan 16. For further information contact: "
                             "johannes.roettenbacher@web.de, Uni Leipzig")
ds = ds.assign_attrs(day="removed")

# group by day and write to outfiles
days, dss = zip(*ds.groupby("time.day"))
paths = [f"202001{d}_FSMETEOR_CHM170158.nc" for d in days]
os.chdir(outpath)
xr.save_mfdataset(dss, paths, format='NETCDF3_CLASSIC')
