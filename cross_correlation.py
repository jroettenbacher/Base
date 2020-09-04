#!/usr/bin/env python
"""Script to compute cross correlation between DSHIP data and LIMRAD94 data to check for possible time shift
input: DSHIP data, LIMRAD94 Doppler velocity
author: Johannes Roettenbacher
"""

import sys
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import datetime as dt
import pyLARDA
import pyLARDA.helpers as h
import logging
import functions_jr as jr
import numpy as np
import pandas as pd
from scipy.signal import correlate

log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# begin_dt = dt.datetime(2020, 1, 24, 0, 0, 0)
# end_dt = dt.datetime(2020, 1, 24, 23, 59, 59)
begin_dt = dt.datetime(2020, 2, 16, 0, 0, 0)
end_dt = dt.datetime(2020, 2, 16, 23, 59, 59)
plot_range = [0, 'max']
larda = pyLARDA.LARDA().connect('eurec4a')
########################################################################################################################
# read in data
########################################################################################################################
# read in radar doppler velocity
radar_vel = larda.read("LIMRAD94_cn_input", "Vel", [begin_dt, end_dt], plot_range)
data = dict()
for var in ['C1Range', 'C2Range', 'C3Range']:
    data.update({var: larda.read('LIMRAD94', var, [begin_dt, end_dt], plot_range)})
range_bins = jr.get_range_bin_borders(3, data)
# set masked values in var to nan
vel = radar_vel['var']
vel[vel.mask] = np.nan
vel = np.asarray(vel)  # convert to ndarray
# average over height of mean doppler velocity
vel_mean = np.nanmean(vel, axis=1)

# read in DSHIP data
seapath = jr.read_seapath(begin_dt)
# add heave rate to data frame
seapath = jr.calc_heave_rate(seapath)

########################################################################################################################
# select closest seapath values to each radar time step
########################################################################################################################
seapath_ts = seapath.index.values.astype(np.float64) / 10 ** 9  # convert datetime index to seconds since 1970-01-01
id_diff_mins = []  # initialize list for indices of the time steps with minimum difference
means_ls = []  # initialize list for heave rate value closest to each radar time step
for t in radar_vel['ts']:
    id_diff_min = h.argnearest(seapath_ts, t)  # find index of nearest seapath time step to radar time step
    id_diff_mins.append(id_diff_min)
    # get time stamp of closest index
    ts_id_diff_min = seapath.index[id_diff_min]
    try:
        means_ls.append(seapath.loc[ts_id_diff_min])
    except KeyError:
        logging.info('Timestamp out of bounds of heave rate time series')

# concatenate all values into one dataframe with the original header (transpose)
seapath_closest = pd.concat(means_ls, axis=1).T

########################################################################################################################
# interpolate heave rate to radar time
########################################################################################################################
# extract heave rate
heave_rate = seapath_closest['Heave Rate [m/s]']
radar_time = radar_vel['ts']
seapath_time = np.asarray([h.dt_to_ts(t) for t in seapath_closest.index])
heave_rate_rts = np.interp(radar_time, seapath_time, heave_rate.values)

########################################################################################################################
# interpolate over nan values
########################################################################################################################
# check for nan values and interpolate them
# heave rate
if any(np.isnan(heave_rate_rts)):
    idx_not_nan = np.asarray(~np.isnan(heave_rate_rts)).nonzero()[0]
    idx_to_ip = np.arange(len(heave_rate_rts))
    heave_rate_ip = np.interp(idx_to_ip, idx_to_ip[idx_not_nan], heave_rate_rts[idx_not_nan])
else:
    heave_rate_ip = heave_rate_rts

# Doppler velocity
if any(np.isnan(vel_mean)):
    idx_not_nan = np.asarray(~np.isnan(vel_mean)).nonzero()[0]
    idx_to_ip = np.arange(len(vel_mean))
    vel_ip = np.interp(idx_to_ip, idx_to_ip[idx_not_nan], vel_mean[idx_not_nan])
else:
    vel_ip = vel_mean

xcorr = np.correlate(vel_ip, heave_rate_ip, 'full')
n_samples = vel_ip.size
dt_lags = np.arange(1-n_samples, n_samples)
time_shift = float(dt_lags[xcorr.argmax()])/4


