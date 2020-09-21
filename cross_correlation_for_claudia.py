#!/usr/bin/env python
"""Script to compute cross correlation between DSHIP data and radar data to check for possible time shift
input: DSHIP data, radar mean Doppler velocity and time stamps
output: print to console of time shifts detected on given days and with used version
author: Johannes Roettenbacher
"""

import sys
# append path to larda
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import pyLARDA
import time
import logging
import pandas as pd
import datetime
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


def read_device_action_log(path="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP",
                           **kwargs):
    """Read in the device action log

    Args:
        path (str): path to file
        **kwargs:
            begin_dt (datetime.datetime): start of file, keep only rows after this date
            end_dt (datetime.datetime): end of file, keep only rows before that date
    Returns: Data frame with the device action log

    """
    # TODO: change filepath
    begin_dt = kwargs['begin_dt'] if 'begin_dt' in kwargs else datetime.datetime(2020, 1, 18)
    end_dt = kwargs['end_dt'] if 'end_dt' in kwargs else datetime.datetime(2020, 3, 1)
    # Action Log, read in action log of CTD actions
    rv_meteor_action = pd.read_csv(f"{path}/20200117-20200301_RV-Meteor_device_action_log.dat", encoding='windows-1252',
                                   sep='\t')
    # set index to date column for easier indexing
    rv_meteor_action.index = pd.to_datetime(rv_meteor_action["Date/Time (Start)"], format="%Y/%m/%d %H:%M:%S")
    rv_meteor_action = rv_meteor_action.loc[begin_dt:end_dt]

    return rv_meteor_action


def read_seapath(date, path="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP",
                 **kwargs):
    """
    Read in Seapath measurements from ship from .dat files to a pandas.DataFrame
    Args:
        date (datetime.datetime): object with date of current file
        path (str): path to seapath files
        **kwargs for read_csv

    Returns:
        seapath (DataFrame): DataFrame with Seapath measurements

    """
    # Seapath attitude and heave data 1Hz
    start = time.time()
    # unpack kwargs
    nrows = kwargs['nrows'] if 'nrows' in kwargs else None
    skiprows = kwargs['skiprows'] if 'skiprows' in kwargs else (1, 2)
    # TODO: change name of file according to your file name
    file = f"{date:%Y%m%d}_DSHIP_seapath_1Hz.dat"
    # set encoding and separator, skip the rows with the unit and type of measurement
    seapath = pd.read_csv(f"{path}/{file}", encoding='windows-1252', sep="\t", skiprows=skiprows,
                          index_col='date time', nrows=nrows)
    # transform index to datetime
    seapath.index = pd.to_datetime(seapath.index, infer_datetime_format=True)
    seapath.index.name = 'datetime'
    seapath.columns = ['Heading [°]', 'Heave [m]', 'Pitch [°]', 'Roll [°]']  # rename columns
    logger.info(f"Done reading in Seapath data in {time.time() - start:.2f} seconds")
    return seapath


def calc_heave_rate(seapath, x_radar=-11, y_radar=4.07, z_radar=15.8, only_heave=False, use_cross_product=True,
                    transform_to_earth=True):
    """
    Calculate heave rate at a certain location of a ship with the measurements of the INS
    Args:
        seapath (pd.DataFrame): Data frame with heading, roll, pitch and heave as columns
        x_radar (float): x position of location with respect to INS in meters
        y_radar (float): y position of location with respect to INS in meters
        z_radar (float): z position of location with respect to INS in meters
        only_heave (bool): whether to use only heave to calculate the heave rate or include pitch and roll induced heave
        use_cross_product (bool): whether to use the cross product like Hannes Griesche https://doi.org/10.5194/amt-2019-434
        transform_to_earth (bool): transform cross product to earth coordinate system as described in https://repository.library.noaa.gov/view/noaa/17400

    Returns:
        seapath (pd.DataFrame): Data frame as input with additional columns radar_heave, pitch_heave, roll_heave and
                                "Heave Rate [m/s]"

    """
    # TODO: adjust x, y, z of your radar
    t1 = time.time()
    logger.info("Calculating Heave Rate...")
    # angles in radians
    pitch = np.deg2rad(seapath["Pitch [°]"])
    roll = np.deg2rad(seapath["Roll [°]"])
    yaw = np.deg2rad(seapath["Heading [°]"])
    # time delta between two time steps in seconds
    d_t = np.ediff1d(seapath.index).astype('float64') / 1e9
    if not use_cross_product:
        logger.info("using a simple geometric approach")
        if not only_heave:
            logger.info("using also the roll and pitch induced heave")
            pitch_heave = x_radar * np.tan(pitch)
            roll_heave = y_radar * np.tan(roll)

        elif only_heave:
            logger.info("using only the ships heave")
            pitch_heave = 0
            roll_heave = 0

        # sum up heave, pitch induced and roll induced heave
        seapath["radar_heave"] = seapath["Heave [m]"] + pitch_heave + roll_heave
        # add pitch and roll induced heave to data frame to include in output for quality checking
        seapath["pitch_heave"] = pitch_heave
        seapath["roll_heave"] = roll_heave
        # ediff1d calculates the difference between consecutive elements of an array
        # heave difference / time difference = heave rate
        heave_rate = np.ediff1d(seapath["radar_heave"]) / d_t

    else:
        logger.info("using the cross product approach from Hannes Griesche")
        # change of angles with time
        d_roll = np.ediff1d(roll) / d_t  # phi
        d_pitch = np.ediff1d(pitch) / d_t  # theta
        d_yaw = np.ediff1d(yaw) / d_t  # psi
        seapath_heave_rate = np.ediff1d(seapath["Heave [m]"]) / d_t  # heave rate at seapath
        pos_radar = np.array([x_radar, y_radar, z_radar])  # position of radar as a vector
        ang_rate = np.array([d_roll, d_pitch, d_yaw]).T  # angle velocity as a matrix
        pos_radar_exp = np.tile(pos_radar, (ang_rate.shape[0], 1))  # expand to shape of ang_rate
        cross_prod = np.cross(ang_rate, pos_radar_exp)  # calculate cross product

        if transform_to_earth:
            logger.info("Transform into Earth Coordinate System")
            phi, theta, psi = roll, pitch, yaw
            a1 = np.cos(theta) * np.cos(psi)
            a2 = -1 * np.cos(phi) * np.sin(psi) + np.sin(theta) * np.cos(psi) * np.sin(phi)
            a3 = np.sin(phi) * np.sin(psi) + np.cos(phi) * np.sin(theta) * np.cos(psi)
            b1 = np.cos(theta) * np.sin(psi)
            b2 = np.cos(phi) * np.cos(psi) + np.sin(theta) * np.sin(phi) * np.sin(psi)
            b3 = -1 * np.cos(psi) * np.sin(phi) + np.cos(phi) * np.sin(theta) * np.sin(psi)
            c1 = -1 * np.sin(theta)
            c2 = np.cos(theta) * np.sin(phi)
            c3 = np.cos(theta) * np.cos(phi)
            Q_T = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])
            # remove first entry of Q_T to match dimension of cross_prod
            Q_T = Q_T[:, :, 1:]
            cross_prod = np.einsum('ijk,kj->kj', Q_T, cross_prod)

        heave_rate = seapath_heave_rate + cross_prod[:, 2]  # calculate heave rate

    # add heave rate to seapath data frame
    # the first calculated heave rate corresponds to the second time step
    heave_rate = pd.DataFrame({'Heave Rate [m/s]': heave_rate}, index=seapath.index[1:])
    seapath = seapath.join(heave_rate)

    logger.info(f"Done with heave rate calculation in {time.time() - t1:.2f} seconds")
    return seapath


def argnearest(array, value):
    """find the index of the nearest value in a sorted array
    for example time or range axis

    Args:
        array (np.array): sorted array with values, list will be converted to 1D array
        value: value to find
    Returns:
        index
    """
    if type(array) == list:
        array = np.array(array)
    i = np.searchsorted(array, value) - 1

    if not i == array.shape[0] - 1:
            if np.abs(array[i] - value) > np.abs(array[i + 1] - value):
                i = i + 1
    return i


def dt_to_ts(dt):
    """datetime to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()


def seconds_to_fstring(time_diff):
    return datetime.datetime.fromtimestamp(time_diff).strftime("%M:%S")


def calc_time_shift_radar_ship(seapath, radar_vel, ts, version=1, **kwargs):
    """Calculate time shift between radar mean Doppler veloctiy and heave rate of ship

    Average the mean Doppler velocity over the whole range and interpolate the heave rate onto the radar time.
    Use a cross correlation method as described on stackoverflow to retrieve the time shift between the two time series.
    Version 1: https://stackoverflow.com/a/13830177
    Version 2: https://stackoverflow.com/a/56432463
    Both versions return the same time shift to the 3rd decimal, when a long enough time series is given, else verion 2
    performs better.

    Args:
        seapath (pd.DataFrame): data frame with heave rate of ship
        radar_vel (ndarray): time x height array of mean Doppler velocity with nan for missing data
        ts (ndarray): time stamps from the radar in unix time (seconds since 1970-01-01)
        version (int): which version to use 1 or 2
        **kwargs:
            plot_xcorr (bool): plot cross correlation function in temporary plot folder

    Returns: time shift in seconds between the two timeseries

    """
    start = time.time()
    # read in kwargs
    plot_xcorr = kwargs['plot_xcorr'] if 'plot_xcorr' in kwargs else False
    begin_dt = kwargs['begin_dt'] if 'begin_dt' in kwargs else seapath.index[0]
    end_dt = kwargs['end_dt'] if 'end_dt' in kwargs else begin_dt + datetime.timedelta(seconds=23 * 60 * 60 + 59 * 60 + 59)
    plot_path = "/projekt1/remsens/work/jroettenbacher/Base/tmp"  # define plot path
    logger.debug(f"plot path: {plot_path}")
    logger.info(f"Start time shift analysis between radar mean Doppler velocity and ship heave rate for "
                f"{begin_dt:%Y-%m-%d}")
    ####################################################################################################################
    # prepare data
    ####################################################################################################################
    # average of mean doppler velocity over height
    vel_mean = np.nanmean(radar_vel, axis=1)

    ####################################################################################################################
    # select closest seapath values to each radar time step
    ####################################################################################################################
    seapath_ts = seapath.index.values.astype(np.float64) / 10 ** 9  # convert datetime index to seconds since 1970-01-01
    id_diff_mins = []  # initialize list for indices of the time steps with minimum difference
    closest_ls = []  # initialize list for heave rate value closest to each radar time step
    for t in ts:
        id_diff_min = argnearest(seapath_ts, t)  # find index of nearest seapath time step to radar time step
        id_diff_mins.append(id_diff_min)
        # get time stamp of closest index
        ts_id_diff_min = seapath.index[id_diff_min]
        try:
            closest_ls.append(seapath.loc[ts_id_diff_min])
        except KeyError:
            logger.warning('Timestamp out of bounds of heave rate time series')

    # concatenate all values into one dataframe with the original header (transpose)
    seapath_closest = pd.concat(closest_ls, axis=1).T

    ####################################################################################################################
    # interpolate heave rate to radar time
    ####################################################################################################################
    # extract heave rate
    heave_rate = seapath_closest['Heave Rate [m/s]']
    radar_time = ts
    seapath_time = np.asarray([dt_to_ts(t) for t in seapath_closest.index])
    heave_rate_rts = np.interp(radar_time, seapath_time, heave_rate.values)  # heave rate at radar time

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
    sr = n / (radar_time[-1] - radar_time[0])  # number of samples / seconds -> sampling rate
    logger.info(f"Using version {version} for cross correlation...")
    if version == 1:
        xcorr = np.correlate(heave_rate_ip, vel_ip, 'full')
        dt_lags = np.arange(1 - n, n)
        time_shift = float(dt_lags[xcorr.argmax()]) / sr
    elif version == 2:
        y2 = heave_rate_ip
        y1 = vel_ip
        xcorr = correlate(y2, y1, mode='same') / np.sqrt(correlate(y1, y1, mode='same')[int(n / 2)] * correlate(y2, y2, mode='same')[int(n / 2)])
        dt_lags = np.linspace(-0.5 * n / sr, 0.5 * n / sr, n)
        time_shift = float(dt_lags[np.argmax(xcorr)])

    if plot_xcorr:
        figname = f"{plot_path}/RV-Meteor_cross_corr_version{version}_mean-V-dop_heave-rate_{begin_dt:%Y-%m-%d_%H%M}-{end_dt:%H%M}.png"
        plt.plot(dt_lags, xcorr)
        plt.xlim((-10, 10))
        plt.ylabel("Cross correlation coefficient")
        plt.xlabel("Artifical Time")
        plt.savefig(figname)
        logger.info(f"Figure saved to: {figname}")
        plt.close()

    logger.debug(f"version: {version}")
    # turn time shift in number of time steps
    sr = 1 / (np.median(np.diff(seapath.index)).astype('float') * 10**-9)  # sampling rate in Hertz
    shift = int(np.round(time_shift * sr))
    logger.info(f"Found time shift to be {time_shift:.3f} seconds")
    logger.info(f"Done with cross correlation, elapsed time = {seconds_to_fstring(time.time() - start)} [min:sec]")
    return time_shift, shift, seapath


if __name__ == "__main__":
    log = logging.getLogger('__main__')
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())

    # read in action log for whole campaign
    action_log = read_device_action_log(begin_dt=datetime.datetime(2020, 1, 17), end_dt=datetime.datetime(2020, 2, 20))
    versions = [1, 2]
    # read in radar data with larda
    # TODO: read in radar data your way
    larda = pyLARDA.LARDA().connect('eurec4a')
    begin_dt = datetime.datetime(2020, 2, 16, 0, 0, 5)
    end_dt = datetime.datetime(2020, 2, 16, 23, 59, 55)
    plot_range = [0, 'max']
    mdv = larda.read("LIMRAD94", "VEL", [begin_dt, end_dt], plot_range)
    # set masked values in var to nan
    radar_vel = mdv['var']
    radar_vel[mdv['mask']] = np.nan
    # extract time stamps
    ts = mdv['ts']
    date = datetime.datetime(2020, 2, 16, 0, 0, 0)
    seapath = read_seapath(date)
    seapath = calc_heave_rate(seapath)
    for version in versions:
        t_shift, shift, seapath = calc_time_shift_radar_ship(seapath, radar_vel, ts, version, plot_xcorr=True)
        print(f"Time shift calculated from {date:%Y-%m-%d}: {t_shift:.4f} with version {version}.")
