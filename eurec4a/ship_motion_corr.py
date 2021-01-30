#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 date  : Fri Nov 20 14:09:09 2020
 author: Claudia Acquistapace
 goal: run ship motion correction code on Meteor data

"""
# importing necessary libraries
import sys

sys.path.append("/projekt1/remsens/work/jroettenbacher/Base/larda")
sys.path.append('.')
import pyLARDA
import pyLARDA.helpers as h
import functions_jr as jr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import time
import logging
from datetime import datetime
from datetime import timedelta
# import atmos
import xarray as xr
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# connect to campaign
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
# generating array of days for the entire campaign
Eurec4aDays = pd.date_range(datetime(2020, 1, 19), datetime(2020, 2, 19), freq='d')
NdaysEurec4a = len(Eurec4aDays)

#######################################################################################
# PARAMETERS TO BE SET BY USERS *

# paths to the different data files and output directories for plots
pathFolderTree = '/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a'
pathFig = f'{pathFolderTree}/ship_motion_correction/plots'
pathNcDataAnc = f'{pathFolderTree}/ship_motion_correction/ncdf_ancillary'

# instrument position coordinates [+5.15m; +5.40m;−15.60m]
r_FMCW = [-11., 4.07, -15.8]  # [m]

# select a date
dayEu = Eurec4aDays[-4]

# extracting strings for yy, dd, mm
yy = str(dayEu)[0:4]
mm = str(dayEu)[5:7]
dd = str(dayEu)[8:10]
date = datetime.strptime(yy + mm + dd, "%Y%m%d")

# selecting time resolution of ship data to be used ( in Hertz)
if date < datetime(2020, 1, 27):
    Hrz = 1
else:
    Hrz = 10

# selecting height for comparison of power spectra of fft transform of w time series
selHeight = 1600.  # height in the chirp 1


#######################################################################################

# definitions of functions necessary for the processing

def find_mdv_time_series(mdv_values, radar_time, NtimeStampsRun):
    """
    author: Claudia Acquistapace, Johannes Roettenbacher
    Identify, given a mean doppler velocity matrix, a sequence of length NtimeStampsRun of values in the matrix
    at a given height that contains the minimum possible amount of nan values in it.

    Args:
        mdv_values (ndarray): time x heigth matrix of Doppler Velocity
        radar_time (ndarray): corresponding radar time stamps in seconds (unix time)
        NtimeStampsRun (int): number of timestamps needed in a mdv series

    Returns:
        valuesTimeSerie (ndarray): time series of Doppler velocity with length NtimeStampsRun
        time_series (ndarray): corresponding time stamps to Doppler velocity time series
        i_height_sel (int): index of chosen height
        valuesColumnMean (ndarray): time series of mean Doppler velocity averaged over height with length NtimeStampsRun

    """
    #  concept: scan the matrix using running mean for every height, and check the number of nans in the selected serie.
    nanAmountMatrix = np.zeros((mdv_values.shape[0] - NtimeStampsRun, mdv_values.shape[1]))
    nanAmountMatrix.fill(np.nan)
    for indtime in range(mdv_values.shape[0] - NtimeStampsRun):
        mdvChunk = mdv_values[indtime:indtime + NtimeStampsRun, :]
        # count number of nans in each height
        nanAmountMatrix[indtime, :] = np.sum(np.isnan(mdvChunk), axis=0)

    # find indeces where nanAmount is minimal
    ntuples = np.where(nanAmountMatrix == np.nanmin(nanAmountMatrix))
    i_time_sel = ntuples[0][0]
    i_height_sel = ntuples[1][0]

    # extract corresponding time series of mean Doppler velocity values for the chirp
    valuesTimeSerie = mdv_values[i_time_sel:i_time_sel + NtimeStampsRun, i_height_sel]
    time_series = radar_time[i_time_sel:i_time_sel + NtimeStampsRun]

    ###### adding test for columns ########
    valuesColumn = mdv_values[i_time_sel:i_time_sel + NtimeStampsRun, :]
    valuesColumnMean = np.nanmean(valuesColumn, axis=1)

    return valuesTimeSerie, time_series, i_height_sel, valuesColumnMean


def f_shiftTimeDataset(dataset):
    """
    author: Claudia Acquistapace
    date: 25 november 2020
    goal : function to shift time variable of the dataset to the central value of the time interval
    of the time step
    input:
        dataset: xarray dataset
    output:
        dataset: xarray dataset with the time coordinate shifted added to the coordinates and the variables now referring to the shifted time array
    """
    # reading time array
    time = dataset['time'].values
    # calculating deltaT using consecutive time stamps
    deltaT = time[2] - time[1]
    # print('delta T for the selected dataset: ', deltaT)
    # defining additional coordinate to the dataset
    dataset.coords['time_shifted'] = dataset['time'] + 0.5 * deltaT
    # exchanging coordinates in the dataset
    datasetNew = dataset.swap_dims({'time': 'time_shifted'})
    return (datasetNew)


def f_calcRMatrix(rollShipArr, pitchShipArr, yawShipArr, NtimeShip):
    """
    author: Claudia Acquistapace
    date : 27/10/2020
    goal: function to calculate R matrix given roll, pitch, yaw
    input:
        rollShipArr: roll array in degrees
        pitchShipArr: pitch array in degrees
        yawShipArr: yaw array in degrees
        NtimeShip: dimension of time array for the definition of R_inv as [3,3,dimTime]
    output:
        R[3,3,Dimtime]: array of rotational matrices, one for each time stamp
    """
    # calculation of the rotational matrix for each time stamp of the ship data for the day
    cosTheta = np.cos(np.deg2rad(rollShipArr))
    senTheta = np.sin(np.deg2rad(rollShipArr))
    cosPhi = np.cos(np.deg2rad(pitchShipArr))
    senPhi = np.sin(np.deg2rad(pitchShipArr))
    cosPsi = np.cos(np.deg2rad(yawShipArr))
    senPsi = np.sin(np.deg2rad(yawShipArr))

    R = np.zeros([3, 3, NtimeShip])
    A = np.zeros([3, 3, NtimeShip])
    B = np.zeros([3, 3, NtimeShip])
    C = np.zeros([3, 3, NtimeShip])
    R.fill(np.nan)
    A.fill(0.)
    B.fill(0.)
    C.fill(0.)

    # indexing for the matrices
    # [0,0]  [0,1]  [0,2]
    # [1,0]  [1,1]  [1,2]
    # [2,0]  [2,1]  [2,2]
    A[0, 0, :] = 1
    A[1, 1, :] = cosTheta
    A[1, 2, :] = -senTheta
    A[2, 1, :] = senTheta
    A[2, 2, :] = cosTheta

    B[0, 0, :] = cosPhi
    B[1, 1, :] = 1
    B[2, 2, :] = cosPhi
    B[0, 2, :] = senPhi
    B[2, 0, :] = -senPhi

    C[0, 0, :] = cosPsi
    C[0, 1, :] = -senPsi
    C[2, 2, :] = 1
    C[1, 0, :] = senPsi
    C[1, 1, :] = cosPsi

    # calculation of the rotation matrix
    A = np.moveaxis(A, 2, 0)
    B = np.moveaxis(B, 2, 0)
    C = np.moveaxis(C, 2, 0)
    R = np.matmul(C, np.matmul(B, A))
    R = np.moveaxis(R, 0, 2)
    return R


def read_seapath(date, path=pathFolderTree + '/instruments/RV-METEOR_DSHIP/',
                 **kwargs):
    """
    Read in daily Seapath measurements from RV Meteor from .dat files to a pandas.DataFrame
    Args:
        date (datetime.datetime): object with date of current file
        path (str): path to seapath files
        kwargs for read_csv
            output_format (str): whether a pandas data frame or a xarray dataset is returned

    Returns:
        seapath (DataFrame): DataFrame with Seapath measurements

    """
    # Seapath attitude and heave data 1 or 10 Hz, choose file depending on date
    start = time.time()
    # unpack kwargs
    nrows = kwargs['nrows'] if 'nrows' in kwargs else None
    skiprows = kwargs['skiprows'] if 'skiprows' in kwargs else (1, 2)
    output_format = kwargs['output_format'] if 'output_format' in kwargs else 'pandas'
    if date < dt.datetime(2020, 1, 27):
        file = f"{date:%Y%m%d}_DSHIP_seapath_1Hz.dat"
    else:
        file = f"{date:%Y%m%d}_DSHIP_seapath_10Hz.dat"
    # set encoding and separator, skip the rows with the unit and type of measurement
    seapath = pd.read_csv(f"{path}/{file}", encoding='windows-1252', sep="\t", skiprows=skiprows,
                          index_col='date time', nrows=nrows)
    # transform index to datetime
    seapath.index = pd.to_datetime(seapath.index, infer_datetime_format=True)
    seapath.index.name = 'time'
    seapath.columns = ['yaw', 'heave', 'pitch', 'roll']  # rename columns
    logger.info(f"Done reading in Seapath data in {time.time() - start:.2f} seconds")
    if output_format == 'pandas':
        pass
    elif output_format == 'xarray':
        seapath = seapath.to_xarray()

    return seapath

def calc_time_shift(w_radar_meanCol, delta_t_min, delta_t_max, resolution, w_ship_chirp, timeSerieRadar, pathFig, chirp,
                    hour, date):
    """
    author: Claudia Acquistapace, Jan. H. Schween, Johannes Roettenbacher
    goal:   calculate and estimation of the time lag between the radar time stamps and the ship time stamp

    NOTE: adding or subtracting the obtained time shift depends on what you did
    during the calculation of the covariances: if you added/subtracted time _shift
    to t_radar you have to do the same for the 'exact time'
    Here is the time shift analysis as plot:
    <ww> is short for <w'_ship*w'_radar> i.e. covariance between vertical speeds from
    ship movements and radar its maximum gives an estimate for optimal agreement in
    vertical velocities of ship and radar
    <Delta w^2> is short for <(w[i]-w[i-1])^2> where w = w_rad - 2*w_ship - this
    is a measure for the stripeness. Its minimum gives an
    estimate how to get the smoothest w data
    Args:
        w_radar_meanCol (ndarray): time series of mean Doppler velocity averaged over height with no nan values
        delta_t_min (float): minimum time shift
        delta_t_max (float): maximum time shift
        resolution (float): time step by which to increment possible time shift
        w_ship_chirp (ndarray): vertical velocity of the radar at the exact chirp time step
        timeSerieRadar (ndarray): time stamps of the mean Doppler velocity time series (w_radar_meanCol)
        pathFig (str): file path where figures should be stored
        chirp (int): which chirp is being processed
        hour (int): which hour of the day is being processed (0-23)
        date (datetime): which day is being processed

    Returns: time shift between radar data and ship data in seconds, quicklooks for each calculation
    """
    labelsizeaxes = 12
    fontSizeTitle = 12
    fontSizeX = 12
    fontSizeY = 12
    plt.gcf().subplots_adjust(bottom=0.15)

    # calculating variation for w_radar
    w_prime_radar = w_radar_meanCol - np.nanmean(w_radar_meanCol)

    # calculating covariance between w-ship and w_radar where w_ship is shifted for each deltaT given by DeltaTimeShift
    DeltaTimeShift = np.arange(delta_t_min, delta_t_max, step=resolution)
    cov_ww = np.zeros(len(DeltaTimeShift))
    deltaW_ship = np.zeros(len(DeltaTimeShift))

    for i in range(len(DeltaTimeShift)):
        # calculate w_ship interpolating it on the new time array (timeShip+deltatimeShift(i))
        T_corr = timeSerieRadar + DeltaTimeShift[i]

        # interpolating w_ship on the shifted time series
        cs_ship = CubicSpline(timeSerieRadar, w_ship_chirp)
        w_ship_shifted = cs_ship(T_corr)

        # calculating w_prime_ship with the new interpolated series
        w_ship_prime = w_ship_shifted - np.nanmean(w_ship_shifted)

        # calculating covariance of the prime series
        cov_ww[i] = np.nanmean(w_ship_prime * w_prime_radar)

        # calculating sharpness deltaW_ship
        w_corrected = w_radar_meanCol - w_ship_shifted
        delta_w = (np.ediff1d(w_corrected)) ** 2
        deltaW_ship[i] = np.nanmean(delta_w)

    # calculating max of covariance and min of deltaW_ship
    minDeltaW = np.nanmin(deltaW_ship)
    indMin = np.where(deltaW_ship == minDeltaW)
    maxCov_w = np.nanmax(cov_ww)
    indMax = np.where(cov_ww == maxCov_w)
    try:
        print(f'Time shift found for chirp {chirp} at hour {hour}: {DeltaTimeShift[indMin][0]}')
        # calculating time shift for radar data
        timeShift_chirp = DeltaTimeShift[indMin][0]

        # plot results
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        fig.tight_layout()
        ax = plt.subplot(1, 1, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        matplotlib.rc('xtick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
        matplotlib.rc('ytick', labelsize=labelsizeaxes)  # sets dimension of ticks in the plots
        ax.plot(DeltaTimeShift, cov_ww, color='red', linestyle=':', label='cov_ww')
        ax.axvline(x=DeltaTimeShift[indMax], color='red', linestyle=':', label='max cov_w')
        ax.plot(DeltaTimeShift, deltaW_ship, color='red', label='Deltaw^2')
        ax.axvline(x=DeltaTimeShift[indMin], color='red', label='min Deltaw^2')
        ax.legend(frameon=False)
        # ax.xaxis_date()
        ax.set_ylim(-0.1, 2.)  # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
        ax.set_xlim(delta_t_min, delta_t_max)  # limits of the x-axes
        ax.set_title(
            f'Covariance and Sharpiness for chirp {chirp}: {date:%Y-%m-%d} hour: {hour}, '
            f'time lag found : {DeltaTimeShift[indMin]}',
            fontsize=fontSizeTitle, loc='left')
        ax.set_xlabel("Time Shift [seconds]", fontsize=fontSizeX)
        ax.set_ylabel('w [m s$^{-1}$]', fontsize=fontSizeY)
        fig.tight_layout()
        fig.savefig(f'{pathFig}/{date:%Y%m%d}_timeShiftQuicklook_chirp{chirp}_hour{hour}.png', format='png')
        plt.close()
    except IndexError:
        print(f'Not enough data points for time shift calculation in chirp {chirp} at hour {hour}!')
        timeShift_chirp = 0

    return timeShift_chirp


def f_calcFftSpectra(vel, time):
    """
    author: Claudia Acquistapace
    goal  :function to calculate fft spectra of a velocity time series
    date  : 07.12.2020
    Parameters
    ----------
    vel : TYPE ndarray [m/s]
        DESCRIPTION: time serie of velocity
    time : TYPE float
        DESCRIPTION. time array corresponding to the velocity in seconds since

    Returns
    -------
    w_pow power spectra obtained with fft transform of the velocity time serie
    freq  corresponding frequencies

    """
    import numpy as np
    w_fft = np.fft.fft(vel)
    N = len(w_fft)
    T_len = (time[-1] - time[0])
    w_pow = (abs(w_fft)) ** 2
    w_pow = w_pow[1:int(N / 2) + 1]
    freq = np.arange(int(N / 2)) * 1 / T_len
    return (w_pow, freq)


def calc_fft_spectra(vel, time):
    """
    Calculate the FFT power spectra of a velocity time series
    Args:
        vel (ndarray): time series of velocities without nan values
        time (ndarray): corresponding time stamps in seconds

    Returns: power spectrum and frequencies of velocity time series for plotting

    """
    from scipy.fft import rfft, rfftfreq
    w_pow = np.abs(rfft(vel))
    N = len(vel)
    freq = rfftfreq(N, 1 / np.mean(np.diff(time)))
    return w_pow, freq


def tick_function(X):
    V = 1 / (X)
    return ["%.f" % z for z in V]


def calc_chirp_timestamps(radar_ts, date, version):
    """ Calculate the exact timestamp for each chirp corresponding with the center or start of the chirp
    The timestamp in the radar file corresponds to the end of a chirp sequence with an accuracy of 0.1 s

    Args:
        radar_ts (ndarray): timestamps of the radar with milliseconds in seconds
        date (datetime.datetime): date which is being processed
        version (str): should the timestamp correspond to the 'center' or the 'start' of the chirp

    Returns: dict with chirp timestamps

    """
    # make lookup table for chirp durations for each chirptable (see projekt1/remsens/hardware/LIMRAD94/chirptables)
    chirp_durations = pd.DataFrame({"Chirp_No": (1, 2, 3), "tradewindCU": (1.022, 0.947, 0.966),
                                    "Doppler1s": (0.239, 0.342, 0.480), "Cu_small_Tint": (0.225, 0.135, 0.181),
                                    "Cu_small_Tint2": (0.562, 0.572, 0.453)})
    # calculate start time of each chirp by subtracting the duration of the later chirp(s) + the chirp itself
    # the timestamp then corresponds to the start of the chirp
    # select chirp durations according to date
    if date < datetime(2020, 1, 29, 18, 0, 0):
        chirp_dur = chirp_durations["tradewindCU"]
    elif date < datetime(2020, 1, 30, 15, 3, 0):
        chirp_dur = chirp_durations["Doppler1s"]
    elif date < datetime(2020, 1, 31, 22, 28, 0):
        chirp_dur = chirp_durations["Cu_small_Tint"]
    else:
        chirp_dur = chirp_durations["Cu_small_Tint2"]

    chirp_timestamps = dict()
    if version == 'center':
        chirp_timestamps["chirp_1"] = radar_ts - chirp_dur[0] - chirp_dur[1] - chirp_dur[2] / 2
        chirp_timestamps["chirp_2"] = radar_ts - chirp_dur[1] - chirp_dur[2] / 2
        chirp_timestamps["chirp_3"] = radar_ts - chirp_dur[2] / 2
    else:
        chirp_timestamps["chirp_1"] = radar_ts - chirp_dur[0] - chirp_dur[1] - chirp_dur[2]
        chirp_timestamps["chirp_2"] = radar_ts - chirp_dur[1] - chirp_dur[2]
        chirp_timestamps["chirp_3"] = radar_ts - chirp_dur[2]

    return chirp_timestamps


def calc_heave_rate_claudia(data, x_radar=-11, y_radar=4.07, z_radar=-15.8):
    """Calculate heave rate at a certain location on a ship according to Claudia Acquistapace's approach

    Args:
        data (xr.DataSet): Data Set with heading, roll, pitch and heave as columns
        x_radar (float): x position of location with respect to INS in meters
        y_radar (float): y position of location with respect to INS in meters
        z_radar (float): z position of location with respect to INS in meters

    Returns: xr.DataSet with additional variable heave_rate

    """
    r_radar = [x_radar, y_radar, z_radar]
    # calculation of w_ship
    heave = data['heave'].values
    timeShip = data['time_shifted'].values.astype('float64') / 10 ** 9
    w_ship = np.diff(heave, prepend=np.nan) / np.diff(timeShip, prepend=np.nan)

    # calculating rotational terms
    roll = data['roll'].values
    pitch = data['pitch'].values
    yaw = data['yaw'].values
    NtimeShip = len(timeShip)
    r_ship = np.zeros((3, NtimeShip))

    # calculate the position of the  radar on the ship r_ship:
    R = f_calcRMatrix(roll, pitch, yaw, NtimeShip)
    for i in range(NtimeShip):
        r_ship[:, i] = np.dot(R[:, :, i], r_radar)

    # calculating vertical component of the velocity of the radar on the ship (v_rot)
    w_rot = np.diff(r_ship[2, :], prepend=np.nan) / np.diff(timeShip, prepend=np.nan)

    # calculating total ship velocity at radar
    heave_rate = w_rot + w_ship
    data['w_rot'] = (('time_shifted'), w_rot)
    data['heave_rate'] = (('time_shifted'), w_ship)
    data['heave_rate_radar'] = (('time_shifted',), heave_rate)

    return data


def calc_shifted_chirp_timestamps(radar_ts, radar_mdv, chirp_ts, rg_borders_id, n_ts_run, Cs_w_radar, **kwargs):
    """
    Calculates the time shift between each chirp time stamp and the ship time stamp for every hour and every chirp.
    Args:
        radar_ts (ndarray): radar time stamps in seconds (unix time)
        radar_mdv (ndarray): time x height matrix of mean Doppler velocity from radar
        chirp_ts (ndarray): exact chirp time stamps
        rg_borders_id (ndarray): indices of chirp boundaries
        n_ts_run (int): number of time steps necessary for mean Doppler velocity time series
        Cs_w_radar (scipy.interpolate.CubicSpline): function of vertical velocity of radar against time
        **kwargs:
            no_chirps (int): number of chirps in radar measurement
            plot_fig (bool): plot quicklook

    Returns: time shifted chirp time stamps, array with time shifts for each chirp and hour

    """
    no_chirps = kwargs['no_chirps'] if 'no_chirps' in kwargs else 3
    delta_t_min = kwargs['delta_t_min'] if 'delta_t_min' in kwargs else radar_ts[0] - radar_ts[1]
    delta_t_max = kwargs['delta_t_max'] if 'delta_t_max' in kwargs else radar_ts[1] - radar_ts[0]
    resolution = kwargs['resolution'] if 'resolution' in kwargs else 0.05
    pathFig = kwargs['pathFig'] if 'pathFig' in kwargs else "./tmp"
    date = kwargs['date'] if 'date' in kwargs else pd.to_datetime(radar_ts[0], unit='s')
    plot_fig = kwargs['plot_fig'] if 'plot_fig' in kwargs else False

    time_shift_array = np.zeros((len(radar_ts), no_chirps))
    chirp_ts_shifted = chirp_ts
    # get total hours in data and then loop through each hour
    hours = np.int(np.ceil(radar_ts.shape[0] * np.mean(np.diff(radar_ts)) / 60 / 60))
    idx = np.int(np.floor(len(radar_ts) / hours))
    for i in range(hours):
        start_idx = i * idx
        if i < hours-1:
            end_idx = (i + 1) * idx
        else:
            end_idx = time_shift_array.shape[0]
        for j in range(no_chirps):
            # set time and range slice
            ts_slice, rg_slice = slice(start_idx, end_idx), slice(rg_borders_id[j], rg_borders_id[j + 1])
            mdv_slice = radar_mdv[ts_slice, rg_slice]
            time_slice = chirp_ts[f'chirp_{j + 1}'][
                ts_slice]  # select the corresponding exact chirp time for the mdv slice
            mdv_series, time_mdv_series, height_id, mdv_mean_col = find_mdv_time_series(mdv_slice, time_slice,
                                                                                        n_ts_run)

            # selecting w_radar values of the chirp over the same time interval as the mdv_series
            w_radar_chirpSel = Cs_w_radar(time_mdv_series)

            # calculating time shift for the chirp and hour if at least n_ts_run measurements are available
            if np.sum(~np.isnan(mdv_mean_col)) == n_ts_run:
                time_shift_array[ts_slice, j] = calc_time_shift(mdv_mean_col, delta_t_min, delta_t_max, resolution,
                                                                w_radar_chirpSel, time_mdv_series,
                                                                pathFig, j + 1, i, date)

            # recalculate exact chirp time including time shift due to lag
            chirp_ts_shifted[f'chirp_{j + 1}'][ts_slice] = chirp_ts[f'chirp_{j + 1}'][ts_slice] - time_shift_array[
                ts_slice, j]
            # get w_radar at the time shifted exact chirp time stamps
            w_radar_exact = Cs_w_radar(chirp_ts_shifted[f'chirp_{j + 1}'][ts_slice])

            if plot_fig:
                # plot mdv time series and shifted radar heave rate
                ts_idx = [h.argnearest(chirp_ts_shifted[f'chirp_{j + 1}'][ts_slice], t) for t in time_mdv_series]
                plot_time = pd.to_datetime(time_mdv_series, unit='s')
                plot_df = pd.DataFrame(dict(time=plot_time, mdv_mean_col=mdv_mean_col,
                                            w_radar_org=Cs_w_radar(time_mdv_series),
                                            w_radar_chirpSel=w_radar_chirpSel,
                                            w_radar_exact_shifted=w_radar_exact[ts_idx])).set_index('time')
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
                ax.plot(plot_df['mdv_mean_col'], color='red', label='mean mdv over column at original radar time')
                ax.plot(plot_df['w_radar_org'], color='blue', linewidth=0.2, label='w_radar at original radar time')
                ax.plot(plot_df['w_radar_chirpSel'], color='blue', label='w_radar at original chirp time')
                ax.plot(plot_df['w_radar_exact_shifted'], '.', color='green', label='w_radar shifted')
                ax.set_ylim(-4., 2.)
                ax.legend(frameon=False)
                # limits of the y-axesn  cmap=plt.cm.get_cmap("viridis", 256)
                ax.set_title(
                    f'Velocity for Time Delay Calculations : {date:%Y-%m-%d} shift = {time_shift_array[start_idx, j]}',
                    loc='left')
                ax.set_xlabel("Time [day hh:mm]")
                ax.set_ylabel('w [m s$^{-1}$]')
                ax.xaxis_date()
                ax.grid()
                fig.autofmt_xdate()
                fig.savefig(f'{pathFig}/{date:%Y%m%d}_time-series_mdv_w-radar_chirp{j + 1}_hour{i}.png')
                plt.close()

    return chirp_ts_shifted, time_shift_array


def calc_corr_matrix_claudia(radar_ts, radar_rg, rg_borders_id, chirp_ts_shifted, Cs_w_radar):
    """
    Calculate the correction matrix to correct the mean Doppler velocity for the ship vertical motion.
    Args:
        radar_ts (ndarray): original radar time stamps in seconds (unix time)
        radar_rg (ndarray): radar range gates
        rg_borders_id (ndarray): indices of chirp boundaries
        chirp_ts_shifted (dict): hourly shifted chirp time stamps
        Cs_w_radar (scipy.interpolate.CubicSpline): function of vertical velocity of radar against time

    Returns: correction matrix for mean Doppler velocity

    """
    no_chirps = len(chirp_ts_shifted)
    corr_matrix = np.zeros((len(radar_ts), len(radar_rg)))
    # get total hours in data and then loop through each hour
    hours = np.int(np.ceil(radar_ts.shape[0] * np.mean(np.diff(radar_ts)) / 60 / 60))
    # divide the day in equal hourly slices
    idx = np.int(np.floor(len(radar_ts) / hours))
    for i in range(hours):
        start_idx = i * idx
        if i < hours-1:
            end_idx = (i + 1) * idx
        else:
            end_idx = len(radar_ts)
        for j in range(no_chirps):
            # set time and range slice
            ts_slice, rg_slice = slice(start_idx, end_idx), slice(rg_borders_id[j], rg_borders_id[j + 1])
            # get w_radar at the time shifted exact chirp time stamps
            w_radar_exact = Cs_w_radar(chirp_ts_shifted[f'chirp_{j + 1}'][ts_slice])
            # add a dimension to w_radar_exact and repeat it over this dimension (range) to fill the hour and
            # chirp of the correction array
            tmp = np.repeat(np.expand_dims(w_radar_exact, 1), rg_borders_id[j + 1] - rg_borders_id[j], axis=1)
            corr_matrix[ts_slice, rg_slice] = tmp

    return corr_matrix


def calc_chirp_int_time(MaxVel, freq, avg_num):
    """
    Calculate the integration time for each chirp
    Args:
        MaxVel (ndarray): Nyquist velocity of each chirp
        freq (ndarray): radar frequency in GHz
        avg_num (ndarray): number of chirps averaged for each chirp measurement

    Returns: ndarray with integration time for each chirp

    """
    chirp_rep_freq = (4 * MaxVel * freq * 10 ** 9) / 299792458  # speed of light
    chirp_duration = 1 / chirp_rep_freq
    chirp_int_time = chirp_duration * avg_num

    return chirp_int_time


def roll_mean_2D(matrix, windowsize, direction):
    """
    Calculate a rolling mean over a given axis of a 2D array
    Args:
        matrix (ndarray): 2D matrix
        windowsize (int): size of the moving window
        direction (str): over which axis to apply the mean, 'row' or 'column'

    Returns: 2D matrix of averaged values

    """
    axis = 0 if direction == 'row' else 1
    df = pd.DataFrame(matrix)  # turn matrix into data frame to use pandas rolling function
    df_roll = df.rolling(window=windowsize, center=True, axis=axis).apply(lambda x: np.nanmean(x))

    return df_roll.values


def plot_fft_spectra(mdv, chirp_ts, mdv_cor, chirp_ts_shifted, mdv_cor_roll, no_chirps, rg_borders_id, n_ts_run,
                     seapath, **kwargs):
    """
    Plot the FFT power spectra of the uncorrected and corrected mean Doppler velocities for each hour and each chirp
    Args:
        mdv (ndarray): original radar mean Doppler velocity
        chirp_ts (ndarray): original exact chirp time stamps
        mdv_cor (ndarray): radar mean Doppler velocity corrected for heave motion
        chirp_ts_shifted (ndarray): time shift corrected exact chirp time stamps
        mdv_cor_roll (ndarry): corrected mean Doppler velocity averaged with a rolling mean over time
        no_chirps (int): number of chirps in radar sample
        rg_borders_id (ndarray): indices of chirp boundaries
        n_ts_run (int): number of time steps necessary for mean Doppler velocity time series
        seapath (xrarray.DataSet): data set with ship motion angles and time shifted to center of each measurement
        **kwargs

    Returns: plot of FFT power spectra of uncorrected and corrected mean Doppler velocity and of ship motion

    """
    date = kwargs['date'] if 'date' in kwargs else seapath['time_shifted'][0]
    pathFig = kwargs['pathFig'] if 'pathFig' in kwargs else './tmp'
    seapath_time = seapath['time'].values.astype(float) / 10**9  # get time in seconds
    dt = np.diff(seapath_time)  # get time resolution
    # calculate angular velocity
    seapath['pitch_rate'] = np.diff(seapath['pitch']) / dt
    seapath['roll_rate'] = np.diff(seapath['roll']) / dt
    seapath = seapath.dropna('time')  # drop nans for interpolation
    seapath_time = seapath['time'].values.astype(float) / 10 ** 9  # get nan free time in seconds
    # prepare interpolation function for angular velocity
    Cs_pitch = CubicSpline(seapath_time, seapath['pitch_rate'])
    Cs_roll = CubicSpline(seapath_time, seapath['roll_rate'])
    Cs_heave = CubicSpline(seapath_time, seapath['heave_rate_radar'])
    # split day in hourly segments
    hours = np.int(np.ceil(chirp_ts.shape[0] * np.mean(np.diff(chirp_ts)) / 60 / 60))
    idx = np.int(np.floor(mdv.shape[0] / hours))
    for i in range(hours):
        start_idx = i * idx
        if i < hours-1:
            end_idx = (i + 1) * idx
        else:
            end_idx = mdv.shape[0]
        for j in range(no_chirps):
            # set time and range slice
            ts_slice, rg_slice = slice(start_idx, end_idx), slice(rg_borders_id[j], rg_borders_id[j + 1])
            mdv_slice = mdv[ts_slice, rg_slice]
            time_slice = chirp_ts[f'chirp_{j + 1}'][ts_slice]  # select the corresponding exact chirp time for the mdv slice
            mdv_slice_cor = mdv_cor[ts_slice, rg_slice]
            time_slice_cor = chirp_ts_shifted[f'chirp_{j + 1}'][ts_slice]
            mdv_slice_cor_roll = mdv_cor_roll[ts_slice, rg_slice]
            # find continuous series of mean doppler velocity for fft analysis
            mdv_series, time_mdv_series, height_id, mdv_mean_col = find_mdv_time_series(mdv_slice, time_slice, n_ts_run)
            mdv_series_cor, time_mdv_series_cor, height_id_cor, mdv_mean_col_cor = find_mdv_time_series(mdv_slice_cor,
                                                                                                        time_slice_cor,
                                                                                                        n_ts_run)
            mdv_series_roll, time_mdv_series_roll, h_id, mdv_mean_roll = find_mdv_time_series(mdv_slice_cor_roll,
                                                                                              time_slice_cor,
                                                                                              n_ts_run)
            # select angular velocities from the ship at the same time steps
            heave_sel = Cs_heave(time_mdv_series)
            pitch_sel = Cs_pitch(time_mdv_series)
            roll_sel = Cs_roll(time_mdv_series)

            if np.sum(~np.isnan(mdv_series)) == n_ts_run:
                # CA fft calculation
                pow_mdv, freq_mdv = f_calcFftSpectra(mdv_series, time_mdv_series)
                pow_mdv_cor, freq_mdv_cor = f_calcFftSpectra(mdv_series_cor, time_mdv_series_cor)
                pow_mdv_roll, freq_mdv_roll = f_calcFftSpectra(mdv_series_roll, time_mdv_series_roll)
                pow_heave, freq_heave = f_calcFftSpectra(heave_sel, time_mdv_series)
                pow_pitch, freq_pitch = f_calcFftSpectra(pitch_sel, time_mdv_series)
                pow_roll, freq_roll = f_calcFftSpectra(roll_sel, time_mdv_series)
                # JR fft calculation
                # pow_mdv, freq_mdv = calc_fft_spectra(mdv_series, time_mdv_series)
                # pow_mdv_cor, freq_mdv_cor = calc_fft_spectra(mdv_series_cor, time_mdv_series_cor)
                # pow_mdv_roll, freq_mdv_roll = calc_fft_spectra(mdv_series_roll, time_mdv_series_roll)
                # pow_heave, freq_heave = calc_fft_spectra(w_radar_sel, time_mdv_series)
                # pow_pitch, freq_pitch = calc_fft_spectra(pitch_sel, time_mdv_series)
                # pow_roll, freq_roll = calc_fft_spectra(roll_sel, time_mdv_series)

                # plot of the power spectra calculated
                fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
                axs[0].loglog(freq_mdv, pow_mdv, label='uncorrected mdv', color='black', alpha=0.5)
                axs[0].loglog(freq_mdv_cor, pow_mdv_cor, label='corrected mdv', color='purple')
                axs[0].loglog(freq_mdv_roll, pow_mdv_roll, label='corrected smoothed mdv', color='pink')
                axs[1].loglog(freq_heave, pow_heave, label='heave rate at radar', color='orange')
                axs[1].loglog(freq_pitch, pow_pitch, label='pitch rate RV Meteor', color='blue')
                axs[1].loglog(freq_roll, pow_roll, label='roll rate RV Meteor', color='red')
                # add second x-axis with period in [s]
                ax2 = axs[0].twiny()
                new_tick_locations = np.array([0.2, 0.1, 0.06666667, 0.05, 0.04, 0.02, 0.01666667])
                ax2.set_xlabel('periods [s]')
                ax2.set_xscale('log')
                ax2.set_xlim(axs[0].get_xlim())
                ax2.set_xticks(new_tick_locations)
                ax2.set_xticklabels(tick_function(new_tick_locations))
                # add grey dashed lines at same place as Hannes Griesche
                for ax in axs:
                    ax.axvline(x=(2 / 2 / np.pi), color='gray', linestyle='--')
                    ax.axvline(x=(1 / 2 / np.pi), color='gray', linestyle='--')
                    ax.axvline(x=(0.1 / 2 / np.pi), color='gray', linestyle='--')
                    ax.set_ylabel("Signal [m$^2$ s$^{-2}$]")
                    ax.set_xlabel("Frequenzy [Hz]")
                    ax.legend()
                    ax.grid()

                fig.suptitle(f"FFT Power Spectrum for {date:%Y%m%d} Chirp {j + 1}, Hour {i}")
                fig.tight_layout()
                fig.savefig(f'{pathFig}/{date:%Y%m%d}_fft_check_CA_chirp{j + 1}_hour{i}.png')
                plt.close()


# %%

print(f'processing date: {date:%Y-%m-%d}')
print('* reading ship data')
ShipDataset = read_seapath(date, output_format='xarray')

# shifting time stamp for ship data hour
ShipDataCenter = f_shiftTimeDataset(ShipDataset)
ShipDataCenter = calc_heave_rate_claudia(ShipDataCenter)
roll = ShipDataCenter['roll'].values
pitch = ShipDataCenter['pitch'].values
yaw = ShipDataCenter['yaw'].values
heave = ShipDataCenter['heave'].values
timeShip = ShipDataCenter['time_shifted'].values.astype('float64') / 10 ** 9
dt = np.round(np.mean(np.diff(timeShip)), 3)
w_ship = ShipDataCenter['heave_rate'].values
w_rot = ShipDataCenter['w_rot'].values
w_radar = ShipDataCenter['heave_rate_radar'].values

# Take from the ship data only the valid times where roll and pitch are not -999 (or nan) - i assume gaps are short and rare
# select valid values of ship time series
i_valid = np.where(~np.isnan(roll) *
                   ~np.isnan(pitch) *
                   ~np.isnan(heave) *
                   ~np.isnan(w_radar))

w_radar_valid = w_radar[i_valid]
timeShip_valid = timeShip[i_valid]
w_ship_valid = w_ship[i_valid]

# plot time series of w_radar and w_ship for the plot interval
plot_df = pd.DataFrame({'time': [h.ts_to_dt(t) for t in timeShip_valid],
                        'w_radar': w_radar_valid,
                        'w_ship': w_ship_valid}).set_index('time')
plot_time_interval = [datetime(2020, 2, 16, 16, 30, 0), datetime(2020, 2, 16, 16, 32, 0)]
plot_df = plot_df.loc[plot_time_interval[0]:plot_time_interval[1]]  # select only 2 minutes of data
fig, ax = plt.subplots()
ax.plot(plot_df['w_radar'], color='red', label='w_radar')
ax.plot(plot_df['w_ship'], color='black', label='w_ship')
ax.legend(frameon=False)
ax.xaxis_date()
fig.autofmt_xdate()
ax.set_ylim(-1.5, 1.5)  # limits of the y-axes
ax.set_title(f'Time series for the day : {date:%Y-%m-%d} - no time shift', loc='left')
ax.set_xlabel("Time [hh:mm:ss]")
ax.set_ylabel('w [m s-1]')
ax.grid()
fig.tight_layout()
fig.savefig(f'{pathFig}/{date:%Y%m%d}_wship_heave_timeSeries.png', format='png')
plt.close()

# %% read in radar data and plot it
time_interval = [date, date + timedelta(0.9999)]  # reading radar data
radarData = larda.read("LIMRAD94", "VEL", time_interval, [0, 'max'])
mdv = radarData['var']
mdv[radarData['mask']] = np.nan
# get the exact chirp time stamps
chirp_ts = calc_chirp_timestamps(radarData['ts'], date, version='center')
for var in ['C1Range', 'C2Range', 'C3Range', 'SeqIntTime', 'DoppLen', 'MaxVel', 'AvgNum', 'RangeRes']:
    # print('loading variable from LV1 :: ' + var)
    radarData.update({var: larda.read("LIMRAD94", var, time_interval, [0, 'max'])})
Nchirps = len(radarData['SeqIntTime']['var'][0])
rg_borders = jr.get_range_bin_borders(3, radarData)
rg_borders_id = rg_borders - np.array(
    [0, 1, 1, 1])  # transform bin boundaries, necessary because python starts counting at 0

# calculate SeqIntTime
radar_aux = xr.open_dataset(radarData['filename'][0])
sampleDur = radar_aux['SampDur']
freq = radar_aux['Freq'].values
MaxVel = radarData['MaxVel']['var'][0].data
avg_num = radarData['AvgNum']['var'][0].data  # number of averaged chirps in each chirp
chirp_int_time = calc_chirp_int_time(MaxVel, freq, avg_num)

# plot on mean doppler velocity time height
fig, ax = pyLARDA.Transformations.plot_timeheight2(radarData, time_interval=plot_time_interval,
                                                   range_interval=[0, 2000])
plt.savefig(f'{pathFig}/{date:%Y%m%d}_mdv_org.png', format='png')
plt.close()

# %% interpolate w_radar onto exact chirp times

Cs = CubicSpline(timeShip_valid, w_radar_valid)  # prepare interpolation of the ship data

# interpolate w_radar for each chirp on the exact time of the chirp
w_radar_chirp = dict()
for i in range(Nchirps):
    w_radar_chirp[f'chirp_{i + 1}'] = Cs(chirp_ts[f'chirp_{i + 1}'])

# %% Caculate time shift between ship and radar for every hour and chirp

delta_t_min = -3.  # minimum time shift
delta_t_max = 3.  # maximum time shift
resolution = 0.05  # step size between min and max delta_t
# calculating time shift for mean doppler velocity proceeding per hour and chirp
# setting the length of the mean doppler velocity time series for calculating time shift
NtimeStampsRun = np.int(10 * 60 / 1.5)  # 10 minutes with time res of 1.5 s

# find a 10 minute mdv time series in every hour of radar data and for each chirp if possible
# calculate time shift for each hour and each chirp
chirp_ts_shifted, time_shift_array = calc_shifted_chirp_timestamps(radarData['ts'], mdv, chirp_ts, rg_borders_id,
                                                                   NtimeStampsRun, Cs,
                                                                   no_chirps=Nchirps, pathFig=pathFig,
                                                                   delta_t_min=delta_t_min, delta_t_max=delta_t_max,
                                                                   date=date, plot_fig=True)
# calculate the correction matrix
corr_matrix = calc_corr_matrix_claudia(radarData['ts'], radarData['rg'], rg_borders_id, chirp_ts_shifted, Cs)

# %%
mdv_corr = mdv + corr_matrix  # calculating corrected mean doppler velocity
# update larda container with new mdv
radarData_cor = h.put_in_container(mdv_corr, radarData)

# %%

# plot of the 2d map of mean doppler velocity corrected for the selected hour
fig, ax = pyLARDA.Transformations.plot_timeheight2(radarData_cor, time_interval=plot_time_interval,
                                                   range_interval=[0, 2000])
fig.savefig(f'{pathFig}/{date:%Y%m%d}_mdv_corr.png')
plt.close()

# applying rolling average to the data
mdv_roll3 = roll_mean_2D(mdv_corr, 3, 'row')
mdv_org_roll3 = roll_mean_2D(mdv, 3, 'row')

# plot of the 2d map of mean doppler velocity corrected for the selected hour with 3 steps running mean applied
radarData_roll = h.put_in_container(mdv_roll3, radarData)
fig, ax = pyLARDA.Transformations.plot_timeheight2(radarData_roll, time_interval=plot_time_interval,
                                                   range_interval=[0, 2000])
fig.savefig(f'{pathFig}/{date:%Y%m%d}_mdv_roll.png')
plt.close()

# plot the mean doppler velocity with 3 steps running mean applied
radarData_org_roll = h.put_in_container(mdv_org_roll3, radarData)
fig, ax = pyLARDA.Transformations.plot_timeheight2(radarData_org_roll, time_interval=plot_time_interval,
                                                   range_interval=[0, 2000])
fig.savefig(f'{pathFig}/{date:%Y%m%d}_mdv_org_roll.png')
plt.close()


# %% plot fft spectra
# calculation of the power spectra of the original and corrected mean Doppler velocity
plot_fft_spectra(mdv, chirp_ts, mdv_corr, chirp_ts_shifted, mdv_roll3, Nchirps, NtimeStampsRun, ShipDataCenter)