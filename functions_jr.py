from itertools import groupby
from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy.signal import correlate
import numpy as np
import pandas as pd
from dask import dataframe as dd
import time
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
from dateutil.relativedelta import relativedelta
import warnings
import logging
import sys
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import pyLARDA
import pyLARDA.helpers as h
from pyLARDA.SpectraProcessing import seconds_to_fstring

logger = logging.getLogger(__name__)


def set_presentation_plot_style():
    plt.style.use('ggplot')
    font = {'family': 'sans-serif', 'size': 24}
    figure = {'figsize': [16, 9], 'dpi': 300}
    plt.rc('font', **font)
    plt.rc('figure', **figure)


# from https://confluence.ecmwf.int/display/COPSRV/CDS+web+API+%28cdsapi%29+training#
def days_of_month(y, m):
    """create a list of days in a month"""
    d0 = dt.datetime(y, m, 1)
    d1 = d0 + relativedelta(months=1)
    out = list()
    while d0 < d1:
        out.append(d0.strftime('%Y-%m-%d'))
        d0 += dt.timedelta(days=1)
    return out


# from https://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results


def daterange(start_date, end_date):
    """ Generator to create a loop over dates by day
    from: https://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python
    :param start_date: datetime object
    :param end_date: datetime object
    :return: loop over date
    """
    for n in range(int((end_date - start_date).days)):
        yield start_date + dt.timedelta(n)


def find_bases_tops(mask, rg_list):
    """
    This function finds cloud bases and tops for a provided binary cloud mask.
    Args:
        mask (np.array, dtype=bool) : bool array containing False = signal, True=no-signal
        rg_list (list) : list of range values

    Returns:
        cloud_prop (list) : list containing a dict for every time step consisting of cloud bases/top indices, range and width
        cloud_mask (np.array) : integer array, containing +1 for cloud tops, -1 for cloud bases and 0 for fill_value
    """
    cloud_prop = []
    cloud_mask = np.full(mask.shape, 0, dtype=np.int)
    for iT in range(mask.shape[0]):
        cloud = [(k, sum(1 for j in g)) for k, g in groupby(mask[iT, :])]
        idx_cloud_edges = np.cumsum([prop[1] for prop in cloud])
        bases, tops = idx_cloud_edges[0:][::2][:-1], idx_cloud_edges[1:][::2]
        if tops.size>0 and tops[-1] == mask.shape[1]:
            tops[-1] = mask.shape[1]-1
        cloud_mask[iT, bases] = -1
        cloud_mask[iT, tops] = +1
        cloud_prop.append({'idx_cb': bases, 'val_cb': rg_list[bases],  # cloud bases
                           'idx_ct': tops, 'val_ct': rg_list[tops],  # cloud tops
                           'width': [ct - cb for ct, cb in zip(rg_list[tops], rg_list[bases])]
                           })
    return cloud_prop, cloud_mask


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
    begin_dt = kwargs['begin_dt'] if 'begin_dt' in kwargs else dt.datetime(2020, 1, 18)
    end_dt = kwargs['end_dt'] if 'end_dt' in kwargs else dt.datetime(2020, 3, 1)
    # Action Log, read in action log of CTD actions
    rv_meteor_action = pd.read_csv(f"{path}/20200117-20200301_RV-Meteor_device_action_log.dat", encoding='windows-1252',
                                   sep='\t', parse_dates=["Date/Time (Start)", "Date/Time (End)"])
    # set index to date column for easier indexing
    rv_meteor_action.index = rv_meteor_action["Date/Time (Start)"]
    rv_meteor_action = rv_meteor_action.loc[begin_dt:end_dt]

    return rv_meteor_action


def read_seapath(date, path="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP",
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
                                "heave_rate"

    """
    t1 = time.time()
    logger.info("Calculating Heave Rate...")
    # angles in radians
    pitch = np.deg2rad(seapath["pitch"])
    roll = np.deg2rad(seapath["roll"])
    yaw = np.deg2rad(seapath["yaw"])
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
        seapath["radar_heave"] = seapath["heave"] + pitch_heave + roll_heave
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
        seapath_heave_rate = np.ediff1d(seapath["heave"]) / d_t  # heave rate at seapath
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
    heave_rate = pd.DataFrame({'heave_rate_radar': heave_rate}, index=seapath.index[1:])
    seapath = seapath.join(heave_rate)

    logger.info(f"Done with heave rate calculation in {time.time() - t1:.2f} seconds")
    return seapath


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


def tick_function(X):
    V = 1 / (X)
    return ["%.f" % z for z in V]


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
    indMin = np.where(deltaW_ship == minDeltaW)[0][0]
    maxCov_w = np.nanmax(cov_ww)
    indMax = np.where(cov_ww == maxCov_w)[0][0]
    try:
        logger.info(f'Time shift found for chirp {chirp} at hour {hour}: {DeltaTimeShift[indMin][0]}')
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
        logger.info(f'Not enough data points for time shift calculation in chirp {chirp} at hour {hour}!')
        timeShift_chirp = 0

    return timeShift_chirp


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
        chirp_ts (dict): original exact chirp time stamps
        mdv_cor (ndarray): radar mean Doppler velocity corrected for heave motion
        chirp_ts_shifted (dict): time shift corrected exact chirp time stamps
        mdv_cor_roll (ndarry): corrected mean Doppler velocity averaged with a rolling mean over time
        no_chirps (int): number of chirps in radar sample
        rg_borders_id (ndarray): indices of chirp boundaries
        n_ts_run (int): number of time steps necessary for mean Doppler velocity time series
        seapath (pandas.DataFrame): data frame with ship motion angles
        **kwargs

    Returns: plot of FFT power spectra of uncorrected and corrected mean Doppler velocity and of ship motion

    """
    date = kwargs['date'] if 'date' in kwargs else seapath.index[0]
    pathFig = kwargs['pathFig'] if 'pathFig' in kwargs else './tmp'
    seapath_time = seapath.index.values.astype(float) / 10**9  # get time in seconds
    dt = np.diff(seapath_time, prepend=np.nan)  # get time resolution
    # calculate angular velocity
    seapath['pitch_rate'] = np.diff(seapath['pitch'], prepend=np.nan) / dt
    seapath['roll_rate'] = np.diff(seapath['roll'], prepend=np.nan) / dt
    seapath = seapath.dropna()  # drop nans for interpolation
    seapath_time = seapath.index.values.astype(float) / 10 ** 9  # get nan free time in seconds
    # prepare interpolation function for angular velocity
    Cs_pitch = CubicSpline(seapath_time, seapath['pitch_rate'])
    Cs_roll = CubicSpline(seapath_time, seapath['roll_rate'])
    Cs_heave = CubicSpline(seapath_time, seapath['heave_rate_radar'])
    # split day in hourly segments
    hours = np.int(np.ceil(chirp_ts['chirp_1'].shape[0] * np.mean(np.diff(chirp_ts['chirp_1'])) / 60 / 60))
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


def get_range_bin_borders(no_chirps, container):
    """get the range bins which correspond to the chirp borders of a FMCW radar

    Args:
        no_chirps (int): Number of chirps
        container (dict): Dictionary with C1/2/3Range variable from LV1 files

    Returns: ndarray with chirp borders including 0
        range_bins

    """
    range_bins = np.zeros(no_chirps + 1, dtype=np.int)  # needs to be length 4 to include all +1 chirp borders
    for i in range(no_chirps):
        try:
            range_bins[i + 1] = range_bins[i] + container[f'C{i + 1}Range']['var'][0].shape
        except ValueError:
            # in case only one file is read in data["C1Range"]["var"] has only one dimension
            range_bins[i + 1] = range_bins[i] + container[f'C{i + 1}Range']['var'].shape

    return range_bins


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
    if date < dt.datetime(2020, 1, 29, 18, 0, 0):
        chirp_dur = chirp_durations["tradewindCU"]
    elif date < dt.datetime(2020, 1, 30, 15, 3, 0):
        chirp_dur = chirp_durations["Doppler1s"]
    elif date < dt.datetime(2020, 1, 31, 22, 28, 0):
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


def calc_heave_corr(container, date, seapath, mean_hr=True):
    """Calculate heave correction for mean Doppler velocity

    Args:
        container (larda container): LIMRAD94 C1/2/3_Range, SeqIntTime, ts, MaxVel, DoppLen
        date (dt.datetime): date of file
        seapath (pd.DataFrame): Data frame with heave rate column ("heave_rate")
        mean_hr (bool): whether to use the mean heave rate over the SeqIntTime or the heave rate at the start time of the chirp

    Returns: heave_corr
        heave_corr (ndarray): heave rate closest to each radar timestep for each height bin, time x range

    """
    start = time.time()
    ####################################################################################################################
    # Calculating Timestamps for each chirp
    ####################################################################################################################
    version = 'start' if mean_hr else 'center'
    chirp_timestamps = calc_chirp_timestamps(container['ts'], date, version)

    # array with range bin numbers of chirp borders
    no_chirps = container['SeqIntTime']['var'].shape[1]
    range_bins = get_range_bin_borders(no_chirps, container)

    seapath_ts = seapath.index.values.astype(np.float64) / 10 ** 9  # convert datetime index to seconds since 1970-01-01
    total_range_bins = range_bins[-1]  # get total number of range bins
    # initialize output variables
    heave_corr = np.empty(shape=(container["ts"].shape[0], total_range_bins))  # time x range
    seapath_out = pd.DataFrame()
    for i in range(no_chirps):
        t1 = time.time()
        # get integration time for chirp
        int_time = pd.Timedelta(seconds=container['SeqIntTime']['var'][0][i])
        # convert timestamps of moments to array
        ts = chirp_timestamps[f"chirp_{i+1}"].values
        id_diff_mins = []  # initialize list for indices of the time steps with minimum difference
        means_ls = []  # initialize list for means over integration time for each radar time step
        for t in ts:
            id_diff_min = h.argnearest(seapath_ts, t)  # find index of nearest seapath time step to radar time step
            id_diff_mins.append(id_diff_min)
            # get time stamp of closest index
            ts_id_diff_min = seapath.index[id_diff_min]
            if mean_hr:
                # select rows from closest time stamp to end of integration time and average, append to list
                means_ls.append(seapath[ts_id_diff_min:ts_id_diff_min+int_time].mean())
            else:
                means_ls.append(seapath.loc[ts_id_diff_min])

        # concatenate all means into one dataframe with the original header (transpose)
        seapath_closest = pd.concat(means_ls, axis=1).T
        # add index with closest seapath time step to radar time step
        seapath_closest.index = seapath.index[id_diff_mins]

        # check if heave rate is greater than 5 standard deviations away from the daily mean and filter those values
        # by averaging the step before and after
        std = np.nanstd(seapath_closest["heave_rate"])
        # try to get indices from values which do not pass the filter. If that doesn't work, then there are no values
        # which don't pass the filter and a ValueError is raised. Write this to a logger
        try:
            id_max = np.asarray(np.abs(seapath_closest["heave_rate"]) > 5 * std).nonzero()[0]
            for j in range(len(id_max)):
                idc = id_max[j]
                warnings.warn(f"Heave rate greater 5 * std encountered ({seapath_closest['heave_rate'][idc]})! \n"
                              f"Using average of step before and after. Index: {idc}", UserWarning)
                # make more sensible filter -> this is a rather sensible filter, because we average over the time
                # steps before and after. Although the values are already averages, this should smooth out outliers
                avg_hrate = (seapath_closest["heave_rate"][idc - 1] + seapath_closest["heave_rate"][idc + 1]) / 2
                if avg_hrate > 5 * std:
                    warnings.warn(f"Heave Rate value greater than 5 * std encountered ({avg_hrate})! \n"
                                  f"Even after averaging step before and after too high value! Index: {idc}",
                                  UserWarning)
                seapath_closest["heave_rate"][idc] = avg_hrate
        except ValueError:
            logger.info(f"All heave rate values are within 5 standard deviation of the daily mean!")

        # add column with chirp number to distinguish in quality control
        seapath_closest["Chirp_no"] = np.repeat(i + 1, len(seapath_closest.index))
        # make data frame with used heave rates
        seapath_out = seapath_out.append(seapath_closest)
        # create array with same dimensions as velocity (time, range)
        heave_rate = np.expand_dims(seapath_closest["heave_rate"].values, axis=1)
        # duplicate the heave correction over the range dimension to add it to all range bins of the chirp
        shape = range_bins[i + 1] - range_bins[i]
        heave_corr[:, range_bins[i]:range_bins[i+1]] = heave_rate.repeat(shape, axis=1)
        logger.info(f"Calculated heave correction for Chirp {i+1} in {time.time() - t1:.2f} seconds")

    logger.info(f"Done with heave correction calculation in {time.time() - start:.2f} seconds")
    return heave_corr, seapath_out


def calc_dopp_res(MaxVel, DoppLen, no_chirps, range_bins):
    """

    Args:
        MaxVel (ndarray): Unambiguous Doppler velocity (+/-) m/s from LV1 file
        DoppLen (ndarray): Number of spectral lines in Doppler spectra from LV1 file
        no_chirps (int): Number of chirps
        range_bins (ndarray): range bin number of lower chirp borders, starts with 0

    Returns: 1D array with Doppler resolution for each height bin

    """
    DoppRes = np.divide(2.0 * MaxVel, DoppLen)
    dopp_res = np.empty(range_bins[-1])
    for ic in range(no_chirps):
        dopp_res[range_bins[ic]:range_bins[ic + 1]] = DoppRes[ic]
    return dopp_res


def heave_rate_to_spectra_bins(heave_corr, doppler_res):
    """translate the heave correction to Doppler spectra bins

    Args:
        heave_corr (ndarray): heave rate closest to each radar timestep for each height bin, time x range
        doppler_res (ndarray): Doppler resolution of each chirp of LIMRAD94 for whole range 1 x range

    Returns: ndarray with number of bins to move each Doppler spectrum
        n_dopp_bins_shift (ndarray): of same dimension as heave_corr

    """
    start = time.time()
    # add a dimension to the doppler_res vector
    doppler_res = np.expand_dims(doppler_res, axis=1)
    # repeat doppler_res to same time dimension as heave_corr
    doppler_res = np.repeat(doppler_res.T, heave_corr.shape[0], axis=0)

    assert doppler_res.shape == heave_corr.shape, f"Arrays have different shape! {doppler_res.shape} " \
                                                  f"and {heave_corr.shape}"

    # calculate number of Doppler bins
    n_dopp_bins_shift = np.round(heave_corr / doppler_res)
    logger.info(f"Done with translation of heave corrections to Doppler bins in {time.time() - start:.2f} seconds")
    return n_dopp_bins_shift, heave_corr


def heave_correction(moments, date, path_to_seapath="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP",
                     mean_hr=True, only_heave=False, use_cross_product=True, transform_to_earth=True, add=False):
    """Correct mean Doppler velocity for heave motion of ship (RV-Meteor)
    Calculate heave rate from seapath measurements and create heave correction array. If Doppler velocity is given as an
    input, correct it and return an array with the corrected Doppler velocities.
    Without Doppler Velocity input, only the heave correction array is returned.

    Args:
        moments: LIMRAD94 moments container as returned by spectra2moments in spec2mom_limrad94.py, C1/2/3_Range,
                 SeqIntTime and Inc_ElA (for time (ts)) from LV1 file
        date (datetime.datetime): object with date of current file
        path_to_seapath (string): path where seapath measurement files (daily dat files) are stored
        mean_hr (bool): whether to use the mean heave rate over the SeqIntTime or the heave rate at the start time of the chirp
        only_heave (bool): whether to use only heave to calculate the heave rate or include pitch and roll induced heave
        use_cross_product (bool): whether to use the cross product like Hannes Griesche https://doi.org/10.5194/amt-2019-434
        transform_to_earth (bool): transform cross product to earth coordinate system as described in https://repository.library.noaa.gov/view/noaa/17400
        add (bool): whether to add the heave rate or subtract it

    Returns: A number of variables
        new_vel (ndarray); corrected Doppler velocities, same shape as moments["VEL"]["var"] or list if no Doppler
        Velocity is given;
        heave_corr (ndarray): heave rate closest to each radar timestep for each height bin, same shape as
        moments["VEL"]["var"];
        seapath_out (pd.DataFrame): data frame with all heave information from the closest time steps to the chirps

    """
    ####################################################################################################################
    # Data Read in
    ####################################################################################################################
    start = time.time()
    logger.info(f"Starting heave correction for {date:%Y-%m-%d}")
    seapath = read_seapath(date, path_to_seapath)

    ####################################################################################################################
    # Calculating Heave Rate
    ####################################################################################################################
    seapath = calc_heave_rate(seapath, only_heave=only_heave, use_cross_product=use_cross_product,
                              transform_to_earth=transform_to_earth)

    ####################################################################################################################
    # Calculating heave correction array and add to Doppler velocity
    ####################################################################################################################
    # make input container to calc_heave_corr function
    container = {'C1Range': moments['C1Range'], 'C2Range': moments['C2Range'], 'C3Range': moments['C3Range'],
                 'SeqIntTime': moments['SeqIntTime'], 'ts': moments['Inc_ElA']['ts']}
    heave_corr, seapath_out = calc_heave_corr(container, date, seapath, mean_hr=mean_hr)

    try:
        if add:
            # create new Doppler velocity by adding the heave rate of the closest time step
            new_vel = moments['VEL']['var'] + heave_corr
        elif not add:
            # create new Doppler velocity by subtracting the heave rate of the closest time step
            new_vel = moments['VEL']['var'] - heave_corr
        # set masked values back to -999 because they also get corrected
        new_vel[moments['VEL']['mask']] = -999
        logger.info(f"Done with heave corrections in {time.time() - start:.2f} seconds")
        return new_vel, heave_corr, seapath_out
    except KeyError:
        logger.info(f"No input Velocities found! Cannot correct Doppler Velocity.\n Returning only heave_corr array!")
        logger.info(f"Done with heave correction calculation only in {time.time() - start:.2f} seconds")
        new_vel = ["I'm an empty list!"]  # create an empty list to return the same number of variables
        return new_vel, heave_corr, seapath_out


def heave_correction_spectra(data, date,
                             path_to_seapath="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP",
                             mean_hr=True, only_heave=False, use_cross_product=True, transform_to_earth=True, add=False,
                             **kwargs):
    """Shift Doppler spectra to correct for heave motion of ship (RV-Meteor)
    Calculate heave rate from seapath measurements and create heave correction array. Translate the heave correction to
    a number spectra bins by which to move each spectra. If Spectra are given, shift them and return a 3D array with the
    shifted spectra.
    Without spectra input, only the heave correction array and the array with the number if bins to move is returned.

    Args:
        data: LIMRAD94 data container filled with spectra and C1/2/3_Range, SeqIntTime, MaxVel, DoppLen from LV1 file
        date (datetime.datetime): object with date of current file
        path_to_seapath (string): path where seapath measurement files (daily dat files) are stored
        mean_hr (bool): whether to use the mean heave rate over the SeqIntTime or the heave rate at the start time of the chirp
        only_heave (bool): whether to use only heave to calculate the heave rate or include pitch and roll induced heave
        use_cross_product (bool): whether to use the cross product like Hannes Griesche https://doi.org/10.5194/amt-2019-434
        transform_to_earth (bool): transform cross product to earth coordinate system as described in https://repository.library.noaa.gov/view/noaa/17400
        add (bool): whether to add the heave rate or subtract it
        **shift (int): number of time steps to shift seapath data

    Returns: A number of variables
        new_spectra (ndarray); corrected Doppler velocities, same shape as data["VHSpec"]["var"] or list if no Doppler
        Spectra are given;
        heave_corr (ndarray): heave rate closest to each radar timestep for each height bin, shape = (time x range);
        seapath_out (pd.DataFrame): data frame with all heave information from the closest time steps to the chirps

    """
    # unpack kwargs
    shift = kwargs['shift'] if 'shift' in kwargs else 0
    ####################################################################################################################
    # Data Read in
    ####################################################################################################################
    start = time.time()
    logger.info(f"Starting heave correction for {date:%Y-%m-%d}")
    seapath = read_seapath(date, path_to_seapath)

    ####################################################################################################################
    # Calculating Heave Rate
    ####################################################################################################################
    seapath = calc_heave_rate(seapath, only_heave=only_heave, use_cross_product=use_cross_product,
                              transform_to_earth=transform_to_earth)

    ####################################################################################################################
    # Use calculated time shift between radar mean doppler velocity and heave rate to shift seapath data
    ####################################################################################################################
    if shift != 0:
        seapath = shift_seapath(seapath, shift)
    else:
        logger.debug(f"Shift is {shift}! Seapath data is not shifted!")

    ####################################################################################################################
    # Calculating heave correction array and translate to number of Doppler bin shifts
    ####################################################################################################################
    # make input container for calc_heave_corr function
    container = {'C1Range': data['C1Range'], 'C2Range': data['C2Range'], 'C3Range': data['C3Range'],
                 'SeqIntTime': data['SeqIntTime'], 'ts': data['VHSpec']['ts'], 'MaxVel': data['MaxVel'],
                 'DoppLen': data["DoppLen"]}
    heave_corr, seapath_out = calc_heave_corr(container, date, seapath, mean_hr=mean_hr)

    no_chirps = len(data['DoppLen'])
    range_bins = get_range_bin_borders(no_chirps, data)
    doppler_res = calc_dopp_res(data['MaxVel'], data['DoppLen'], no_chirps, range_bins)

    n_dopp_bins_shift, heave_corr = heave_rate_to_spectra_bins(heave_corr, doppler_res)

    ####################################################################################################################
    # Shifting spectra and writing to new 3D array
    ####################################################################################################################

    try:
        # correct spectra for heave rate by moving it by the corresponding number of Doppler bins
        spectra = data['VHSpec']['var']
        new_spectra = np.empty_like(spectra)
        for iT in range(data['n_ts']):
            # loop through time steps
            for iR in range(data['n_rg']):
                # loop through range gates
                # TODO: check if mask is True and skip, although masked shifted spectra do not introduce any error,
                # this might speed up things...
                shift = int(n_dopp_bins_shift[iT, iR])
                spectrum = spectra[iT, iR, :]
                if add:
                    new_spec = np.roll(spectrum, shift)
                elif not add:
                    new_spec = np.roll(spectrum, -shift)

                new_spectra[iT, iR, :] = new_spec

        logger.info(f"Done with heave corrections in {time.time() - start:.2f} seconds")
        return new_spectra, heave_corr, n_dopp_bins_shift, seapath_out
    except KeyError:
        logger.info(f"No input spectra found! Cannot shift spectra.\n Returning only heave_corr and n_dopp_bins_shift array!")
        logger.info(f"Done with heave correction calculation only in {time.time() - start:.2f} seconds")
        new_spectra = ["I'm an empty list!"]  # create an empty list to return the same number of variables
        return new_spectra, heave_corr, n_dopp_bins_shift, seapath_out


def calc_sensitivity_curve(program, campaign, rain_flag=True):
    """Calculate statistics of the sensitivity limit over height for specified chirp table (sensitivity curves)

    Args:
        program (list): list of program numbers e.g. 'P07'
        campaign (str): name of campaign from where to load data
        rain_flag (bool): whether to apply a rain flag to the data, currently implemented for eurec4a

    Returns: dictionary of dictionaries with min, max, mean sensitivity curve for horizontal and vertical channel,
    filtered and not filtered with DWD rain flag

    """

    start = time.time()
    programs = ["P09", "P06", "P07", "P03"]  # implemented chirp tables
    for p in program:
        assert p in programs, f"Please use program codes like 'P07' to select chirptable! Not {p}!" \
                              f"Check functions documentation to see which program corresponds to which chirptable"
    # Load LARDA
    larda = pyLARDA.LARDA().connect(campaign, build_lists=True)
    system = "LIMRAD94"

    # get duration of use for each chirp tables
    begin_dts, end_dts = get_chirp_table_durations(program)
    plot_range = [0, 'max']

    # read in sensitivity variables for each chirp table over whole range (all chirps)
    slv = {}
    slh = {}
    rain_flag_dwd = {}
    rain_flag_dwd_ip = {}

    for p in program:
        logger.info(f"Calculating sensitivity curve for program {p}")
        begin_dt = begin_dts[p]
        end_dt = end_dts[p]
        t1 = time.time()
        slh[p] = larda.read(system, "SLh", [begin_dt, end_dt], plot_range)
        slv[p] = larda.read(system, "SLv", [begin_dt, end_dt], plot_range)
        # Note: The sensitivity and the total noise are combined in V-channel. Thus SLV = 4 * SLV - SLH
        # see mattermost LIMRAD94 channel #sensitivity for details
        slv[p]['var'] = 4 * slv[p]['var'] - slh[p]['var']
        logger.info(f"Read in sensitivity limits in {time.time() - t1:.2f} seconds")

        if rain_flag and campaign == 'eurec4a':
            # DWD rain flag
            # weather data, time res = 1 min, only read in Dauer (duration) column, gives rain duration in seconds
            t1 = time.time()
            weather = pd.read_csv("/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DWD/20200114_M161_Nsl.CSV", sep=";",
                                  index_col="Timestamp", usecols=[0, 5], squeeze=True)
            weather.index = pd.to_datetime(weather.index, format="%d.%m.%Y %H:%M")
            weather = weather[begin_dt:end_dt]  # select date range
            rain_flag_dwd[p] = weather > 0  # set rain flag if rain duration is greater 0 seconds
            # interpolate rainflag to radar time resolution
            f = interpolate.interp1d(h.dt_to_ts(rain_flag_dwd[p].index), rain_flag_dwd[p], kind='nearest',
                                     fill_value="extrapolate")
            rain_flag_dwd_ip[p] = f(np.asarray(slv[p]['ts']))
            # adjust rainflag to sensitivity limit dimensions
            rain_flag_dwd_ip[p] = np.tile(rain_flag_dwd_ip[p], (slv[p]['var'].shape[1], 1)).swapaxes(0,1).copy()
            logger.info(f"Read in and interpolated DWD rainflag in {time.time() - t1:.2f} seconds")
        else:
            rain_flag = False

    # get statistics of sensitivity limit for whole period of operation
    t1 = time.time()
    stats = {k: {} for k in ['mean_slv', 'median_slv', 'min_slv', 'max_slv',
                             'mean_slh', 'median_slh', 'min_slh', 'max_slh',
                             'mean_slv_f', 'median_slv_f', 'min_slv_f', 'max_slv_f',
                             'mean_slh_f', 'median_slh_f', 'min_slh_f', 'max_slh_f']}
    for p in program:
        stats['mean_slv'][p] = np.mean(slv[p]['var'], axis=0)
        stats['mean_slh'][p] = np.mean(slh[p]['var'], axis=0)
        stats['median_slv'][p] = np.median(slv[p]['var'], axis=0)
        stats['median_slh'][p] = np.median(slh[p]['var'], axis=0)
        stats['min_slv'][p] = np.min(slv[p]['var'], axis=0)
        stats['min_slh'][p] = np.min(slh[p]['var'], axis=0)
        stats['max_slv'][p] = np.max(slv[p]['var'], axis=0)
        stats['max_slh'][p] = np.max(slh[p]['var'], axis=0)
        if rain_flag:
            # rainflag filtered means
            stats['mean_slv_f'][p] = np.mean(np.ma.masked_where(rain_flag_dwd_ip[p] == 1, slv[p]['var']), axis=0)
            stats['mean_slh_f'][p] = np.mean(np.ma.masked_where(rain_flag_dwd_ip[p] == 1, slh[p]['var']), axis=0)
            stats['median_slv_f'][p] = np.median(np.ma.masked_where(rain_flag_dwd_ip[p] == 1, slv[p]['var']), axis=0)
            stats['median_slh_f'][p] = np.median(np.ma.masked_where(rain_flag_dwd_ip[p] == 1, slh[p]['var']), axis=0)
            stats['min_slv_f'][p] = np.min(np.ma.masked_where(rain_flag_dwd_ip[p] == 1, slv[p]['var']), axis=0)
            stats['min_slh_f'][p] = np.min(np.ma.masked_where(rain_flag_dwd_ip[p] == 1, slh[p]['var']), axis=0)
            stats['max_slv_f'][p] = np.max(np.ma.masked_where(rain_flag_dwd_ip[p] == 1, slv[p]['var']), axis=0)
            stats['max_slh_f'][p] = np.max(np.ma.masked_where(rain_flag_dwd_ip[p] == 1, slh[p]['var']), axis=0)

    logger.info(f"Calculated statistics of sensitivity limits for rain filtered and non filtered data in "
          f"{time.time() - t1:.2f} seconds")
    logger.info(f"Done with calculate_sensitivity_curve in {time.time() - start:.2f} seconds")

    return stats


def get_chirp_table_durations(program):
    """get the begin and end datetime of a specified chirp table

    Implemented chirp tables: "tradewindCU (P09)", "Cu_small_Tint (P06)", "Cu_small_Tint2 (P07)", "Lindenberg (P03)".

    Args:
        program (list): list of program names, e.g. "P07"

    Returns: begin and end datetime of specified chirp table in a dictionary with program as keys

    """
    # define durations of use for each chirp table (program)
    begin_dts = {'P09': dt.datetime(2020, 1, 17, 0, 0, 5), 'P06': dt.datetime(2020, 1, 30, 15, 30, 5),
                 'P07': dt.datetime(2020, 1, 31, 22, 30, 5), 'P03': dt.datetime(2020, 7, 15, 0, 0, 5)}
    end_dts = {'P09': dt.datetime(2020, 1, 27, 0, 0, 5), 'P06': dt.datetime(2020, 1, 30, 23, 42, 0),
               'P07': dt.datetime(2020, 2, 19, 23, 59, 55), 'P03': dt.datetime(2020, 10, 20, 23, 59, 55)}
    for p in program:
        assert p in begin_dts.keys() and p in end_dts.keys(), f"{p} is not a valid/implemented chirp program number"

    # create dictionaries with the wanted program durations only
    begin_out = {p: begin_dts[p] for p in program}
    end_out = {p: end_dts[p] for p in program}

    return begin_out, end_out


def get_chirp_table_names(program):
    """get the names of the supplied chirp table programs

    Implemented chirp tables: "tradewindCU (P09)", "Cu_small_Tint (P06)", "Cu_small_Tint2 (P07)", "Lindenberg (P03)".

    Args:
        program (list): list of chirp program names, e.g. "P07"

    Returns: dictionary with the corresponding program names to each supplied program number

    """
    program_names = {'P09': "tradewindCU (P09)", 'P06': "Cu_small_Tint (P06)", 'P07': "Cu_small_Tint2 (P07)",
                     'P03': "Lindenberg (P03)"}
    for p in program:
        assert p in program_names.keys(), f"{p} is not a valid/implemented chirp program number"

    return {p: program_names[p] for p in program}


def calc_time_shift_limrad_seapath(seapath, version=1, **kwargs):
    """Calculate time shift between LIMRAD94 mean Doppler veloctiy and heave rate of RV Meteor

    Average the mean Doppler velocity over the whole range and interpolate the heave rate onto the radar time.
    Use a cross correlation method as described on stackoverflow to retrieve the time shift between the two time series.
    Version 1: https://stackoverflow.com/a/13830177
    Version 2: https://stackoverflow.com/a/56432463
    Both versions return the same time shift to the 3rd decimal.

    Args:
        seapath (pd.DataFrame): data frame with heave rate of RV-Meteor
        version (int): which version to use 1 or 2
        **kwargs:
            plot_xcorr (bool): plot cross correlation function in temporary plot folder

    Returns: time shift in seconds between the two timeseries

    """
    start = time.time()
    # read in kwargs
    plot_xcorr = kwargs['plot_xcorr'] if 'plot_xcorr' in kwargs else False
    begin_dt = kwargs['begin_dt'] if 'begin_dt' in kwargs else seapath.index[0]
    end_dt = kwargs['end_dt'] if 'end_dt' in kwargs else begin_dt + dt.timedelta(seconds=23 * 60 * 60 + 59 * 60 + 59)
    plot_path = "/projekt1/remsens/work/jroettenbacher/Base/tmp"  # define plot path
    logger.debug(f"plot path: {plot_path}")
    logger.info(f"Start time shift analysis between LIMRAD94 mean Doppler velocity and RV Meteor heave rate for "
                f"{begin_dt:%Y-%m-%d}")
    plot_range = [0, 'max']
    larda = pyLARDA.LARDA().connect('eurec4a')
    ####################################################################################################################
    # read in data
    ####################################################################################################################
    # read in radar doppler velocity
    radar_vel = larda.read("LIMRAD94", "VEL", [begin_dt, end_dt], plot_range)
    # needed if time shift should be calculated for each chirp separately, leave in for now
    # data = dict()
    # for var in ['C1Range', 'C2Range', 'C3Range']:
    #     data.update({var: larda.read('LIMRAD94', var, [begin_dt, end_dt], plot_range)})
    # range_bins = jr.get_range_bin_borders(3, data)

    # set masked values in var to nan
    vel = radar_vel['var']
    vel[radar_vel['mask']] = np.nan
    # average of mean doppler velocity over height
    vel_mean = np.nanmean(vel, axis=1)

    ####################################################################################################################
    # select closest seapath values to each radar time step
    ####################################################################################################################
    seapath_ts = seapath.index.values.astype(np.float64) / 10 ** 9  # convert datetime index to seconds since 1970-01-01
    id_diff_mins = []  # initialize list for indices of the time steps with minimum difference
    closest_ls = []  # initialize list for heave rate value closest to each radar time step
    for t in radar_vel['ts']:
        id_diff_min = h.argnearest(seapath_ts, t)  # find index of nearest seapath time step to radar time step
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
    heave_rate = seapath_closest['heave_rate']
    radar_time = radar_vel['ts']
    seapath_time = np.asarray([h.dt_to_ts(t) for t in seapath_closest.index])
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

    if int(time_shift) == 0:
        # armin doesn't make sense, since the signals are positively correlated
        logger.info(f"Time shift was found to be {time_shift}, trying argmin()")
        time_shift = float(dt_lags[np.argmin(xcorr)])

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


def shift_seapath(seapath, shift):
    """Shift seapath values by given shift

    Args:
        seapath (pd.Dataframe): Dataframe with heave motion of RV-Meteor
        shift (int): number of time steps to shift data

    Returns: shifted Dataframe

    """
    start = time.time()
    logger.info(f"Shifting seapath data by {shift} time steps.")
    # get day of seapath data
    datetime = seapath.index[0]
    # shift seapath data by shift
    seapath_shifted = seapath.shift(periods=shift)

    # replace Nans at start with data from the previous day or from following day
    if shift > 0:
        datetime_previous = datetime - dt.timedelta(1)  # get date of previous day
        skiprows = np.arange(1, len(seapath) - shift + 2)  # define rows to skip on read in
        # read in one more row for heave rate calculation
        seapath_previous = read_seapath(datetime_previous, nrows=shift + 1, skiprows=skiprows)
        seapath_previous = calc_heave_rate(seapath_previous)
        seapath_previous = seapath_previous.iloc[1:, :]  # remove first row (=nan)
        # remove index and replace with index from original data frame
        seapath_previous = seapath_previous.reset_index(drop=True).set_index(seapath_shifted.iloc[0:shift, :].index)
        seapath_shifted.update(seapath_previous)  # overwrite nan values in shifted data frame
    else:
        datetime_following = datetime + dt.timedelta(1)  # get date from following day
        seapath_following = read_seapath(datetime_following, nrows=np.abs(shift))
        seapath_following = calc_heave_rate(seapath_following)
        # overwrite nan values
        # leaves in one NaN value because the heave rate of the first time step of a day cannot be calculated
        # one nan is better than many (shift) though, so this is alright
        seapath_following = seapath_following.reset_index(drop=True).set_index(seapath_shifted.iloc[shift:, :].index)
        seapath_shifted.update(seapath_following)  # overwrite nan values in shifted data frame

    logger.info(f"Done with shifting seapath data, elapsed time = {seconds_to_fstring(time.time() - start)} [min:sec]")
    return seapath_shifted


def read_dship(date, **kwargs):
    """Read in 1 Hz DSHIP data and return pandas DataFrame

    Args:
        date (str): yyyymmdd (eg. 20200210)
        **kwargs: kwargs for pd.read_csv (not all implemented) https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

    Returns: pd.DataFrame with 1 Hz DSHIP data

    """
    tstart = time.time()
    path = kwargs['path'] if 'path' in kwargs else "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP"
    skiprows = kwargs['skiprows'] if 'skiprows' in kwargs else (1, 2)
    nrows = kwargs['nrows'] if 'nrows' in kwargs else None
    cols = kwargs['cols'] if 'cols' in kwargs else None  # always keep the 0th column (datetime column)
    file = f"{path}/RV-Meteor_DSHIP_all_1Hz_{date}.dat"
    # set encoding and separator, skip the rows with the unit and type of measurement, set index column
    df = pd.read_csv(file, encoding='windows-1252', sep="\t", skiprows=skiprows, index_col='date time', nrows=nrows,
                     usecols=cols, na_values='-999.0')
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
    df.index.rename('datetime', inplace=True)

    logger.info(f"Done reading in DSHIP data in {time.time() - tstart:.2f} seconds")

    return df


def find_closest_timesteps(df, ts):
    """Find closest time steps in a dataframe to a time series

    Args:
        df (pd.DataFrame): DataFrame with DatetimeIndex
        ts (ndarray): array with time stamps in unix format (seconds since 1-1-1970)

    Returns: pd.DataFrame with only the closest time steps to ts

    """
    tstart = time.time()
    try:
        assert df.index.inferred_type == 'datetime64', "Dataframe Index is not a DatetimeIndex trying to turn into one"
    except AssertionError:
        df.index = pd.to_datetime(df.index, infer_datetime_format=True)

    df_ts = df.index.values.astype(np.float64) / 10 ** 9  # convert datetime index to seconds since 1970-01-01
    df_list = []  # initialize lsit to append df rows closest to input time steps to
    for t in ts:
        id_diff_min = h.argnearest(df_ts, t)  # find index of nearest dship time step to input time step
        ts_id_diff_min = df.index[id_diff_min]  # get time stamp of closest index
        df_list.append(df.loc[ts_id_diff_min])  # append row to list

    # concatenate all rows into one dataframe with the original header (transpose)
    df_closest = pd.concat(df_list, axis=1).T
    logger.info(f"Done finding closest time steps in {time.time() - tstart:.2f} seconds")

    return df_closest


def read_rainrate(path="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DWD"):
    """Read in rain rate from RV-Meteor during Eurec4a"""
    file = "20200114_M161_Nsl.CSV"
    rr = pd.read_csv(f'{path}/{file}', sep=';', usecols=['Timestamp', '   Dauer', 'RR_WS100_h', 'Nabs_WS100'])
    rr.Timestamp = pd.to_datetime(rr.Timestamp, format="%d.%m.%Y %H:%M")  # turn Timestamp datetime
    rr.set_index('Timestamp', inplace=True)  # set Timestamp column as DateTimeIndex
    rr.columns = ['Dauer', 'RR_WS100_h', 'Nabs_WS100']  # rename columns
    return rr


def add_season_to_df(df):
    """Add a column with season to a dataframe containing a date index

    Args:
        df (pd.DataFrame): dataframe with date index

    Returns: DataFrame as input with new column season

    """
    lookup = {11: 'Winter', 12: 'Winter', 1: 'Winter', 2: 'Spring', 3: 'Spring', 4: 'Spring', 5: 'Summer', 6: 'Summer',
              7: 'Summer', 8: 'Autumn', 9: 'Autumn', 10: 'Autumn'}
    df['date'] = df.index
    df['season'] = df['date'].apply(lambda x: lookup[x.month])
    df = df.drop('date', 1)
    return df


def merge_csv(path, outname, **kwargs):
    """Merge all csv files in a directory to one

    Args:
        path (str): path to directory with csv files
        outname (str): name of the output csv file
        kwargs: keywords taken by read csv
        - sep

    Returns: saves a new csv file, omitting the date in the file name

    """
    sep = kwargs['sep'] if 'sep' in kwargs else ','
    dfs = dd.read_csv(f"{path}/*.csv", sep=sep)
    dfs.to_csv(f"{path}/{outname}", single_file=True)
    logger.info(f"Merged all csv files in {path} and saved to {outname}")


if __name__ == '__main__':
    import sys, time
    import datetime as dt
    sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
    sys.path.append('.')
    import pyLARDA
    import numpy as np
    import logging
    import pandas as pd

    log = logging.getLogger('__main__')
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())

    # # test heave correction
    # larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
    # begin_dt = dt.datetime(2020, 2, 5, 0, 0, 5)
    # end_dt = dt.datetime(2020, 2, 5, 23, 59, 55)
    # plot_range = [0, 'max']
    # mdv = larda.read("LIMRAD94_cn_input", "Vel", [begin_dt, end_dt], plot_range)
    # moments = {"VEL": mdv}
    # for var in ['C1Range', 'C2Range', 'C3Range', 'SeqIntTime', 'Inc_ElA', 'DoppLen', 'MaxVel']:
    #     print('loading variable from LV1 :: ' + var)
    #     moments.update({var: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'])})
    # new_vel, heave_corr, seapath_out = heave_correction(moments, begin_dt, use_cross_product=True)
    # # test without Doppler Velocity input
    # moments.__delitem__('VEL')
    # new_vel, heave_corr, seapath_out = heave_correction(moments, begin_dt, use_cross_product=True)
    # print("Done Testing heave_correction...")

    # # time shift analysis
    # date = dt.datetime(2020, 2, 10)
    # # seapath = read_seapath(date)
    # seapath = calc_heave_rate(seapath)
    # t_shift, shift, seapath = calc_time_shift_limrad_seapath(seapath)
    # seapath_shifted = shift_seapath(seapath, -shift)

    # test read in of DSHIP data
    # date = '20200125'
    # dship = read_dship(date)

    # # test find_closest_timesteps
    # larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
    # begin_dt = dt.datetime(2020, 2, 5, 0, 0, 5)
    # end_dt = dt.datetime(2020, 2, 5, 23, 59, 55)
    # plot_range = [0, 'max']
    # mdv = larda.read("LIMRAD94_cn_input", "Vel", [begin_dt, end_dt], plot_range)
    # ts = mdv["ts"]
    # date = begin_dt.strftime("%Y%m%d")
    # dship = read_dship(date, cols=[0, 5, 6])
    # dship_closest = find_closest_timesteps(dship, ts)

    # test read rain rate
    # rr = read_rainrate()

    # test read action log
    # action_log = read_device_action_log()

    # test merge_csv
    path = "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/virga_sniffer"
    outname = "RV-Meteor_virga-collection_all.csv"
    merge_csv(path, outname, sep=";")
