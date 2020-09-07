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
import matplotlib.pyplot as plt

log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())


def calc_time_shift_limrad_seapath(date, version=1, plot_xcorr=False):
    """Calculate time shift between LIMRAD94 mean Doppler veloctiy and heave rate of RV Meteor

    Average the mean Doppler velocity over the whole range and interpolate the heave rate onto the radar time.
    Use a cross correlation method as described on stackoverflow to retrieve the time shift between the two time series.
    Version 1: https://stackoverflow.com/a/13830177
    Version 2: https://stackoverflow.com/a/56432463
    Both versions return the same time shift to the 3rd decimal.

    Args:
        date (datetime.date): date for which to calculate time shift
        version (int): which version to use 1 or 2
        plot_xcorr (bool): plot cross correlation function in temporary plot folder

    Returns: time shift in seconds between the two timeseries

    """
    plot_path = "/projekt1/remsens/work/jroettenbacher/Base/tmp"
    begin_dt = dt.datetime.combine(date, dt.datetime.min.time())
    end_dt = begin_dt + dt.timedelta(seconds=23 * 60 * 60 + 59 * 60 + 59)
    plot_range = [0, 'max']
    larda = pyLARDA.LARDA().connect('eurec4a')
    ####################################################################################################################
    # read in data
    ####################################################################################################################
    # read in radar doppler velocity
    radar_vel = larda.read("LIMRAD94_cn_input", "Vel", [begin_dt, end_dt], plot_range)
    # needed if time shift should be calculated for each chirp separately, leave in for now
    # data = dict()
    # for var in ['C1Range', 'C2Range', 'C3Range']:
    #     data.update({var: larda.read('LIMRAD94', var, [begin_dt, end_dt], plot_range)})
    # range_bins = jr.get_range_bin_borders(3, data)

    # set masked values in var to nan
    vel = radar_vel['var']
    vel[vel.mask] = np.nan
    vel = np.asarray(vel)  # convert to ndarray
    # average of mean doppler velocity over height
    vel_mean = np.nanmean(vel, axis=1)

    # read in DSHIP data
    seapath = jr.read_seapath(begin_dt)
    # add heave rate to data frame
    seapath = jr.calc_heave_rate(seapath)

    ####################################################################################################################
    # select closest seapath values to each radar time step
    ####################################################################################################################
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

    ####################################################################################################################
    # interpolate heave rate to radar time
    ####################################################################################################################
    # extract heave rate
    heave_rate = seapath_closest['Heave Rate [m/s]']
    radar_time = radar_vel['ts']
    seapath_time = np.asarray([h.dt_to_ts(t) for t in seapath_closest.index])
    heave_rate_rts = np.interp(radar_time, seapath_time, heave_rate.values)

    ####################################################################################################################
    # interpolate over nan values
    ####################################################################################################################
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

    n = vel_ip.size
    sr = n / (24 * 60 * 60)  # number of samples per day / seconds per day -> sampling rate

    if version == 1:
        xcorr = np.correlate(heave_rate_ip, vel_ip, 'full')
        dt_lags = np.arange(1 - n, n)
        time_shift = float(dt_lags[xcorr.argmax()]) / sr
        if plot_xcorr:
            plt.plot(dt_lags, xcorr)
            plt.savefig(f"{plot_path}/RV-Meteor_cross_corr_version1_mean-V-dop_heave-rate_{begin_dt:%Y-%m-%d}.png")
            plt.close()
    elif version == 2:
        y2 = heave_rate_ip
        y1 = vel_ip
        corr = correlate(y2, y1, mode='same') / np.sqrt(correlate(y1, y1, mode='same')[int(n / 2)] * correlate(y2, y2, mode='same')[int(n / 2)])
        delay_array = np.linspace(-0.5 * n / sr, 0.5 * n / sr, n)
        time_shift = float(delay_array[np.argmax(corr)])
        if plot_xcorr:
            plt.plot(delay_array, corr)
            plt.savefig(f"{plot_path}/RV-Meteor_cross_corr_version2_mean-V-dop_heave-rate_{begin_dt:%Y-%m-%d}.png")
            plt.close()
    else:
        print(f"Wrong version selected! {version}")

    return time_shift
