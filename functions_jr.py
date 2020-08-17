from itertools import groupby
from scipy import interpolate
import numpy as np
import pandas as pd
import time
import matplotlib as mpl
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


def read_seapath(date, path="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP"):
    """
    Read in Seapath measurements from RV Meteor from .dat files to a pandas.DataFrame
    Args:
        date (datetime.datetime): object with date of current file
        path (str): path to seapath files

    Returns:
        seapath (DataFrame): DataFrame with Seapath measurements

    """
    # Seapath attitude and heave data 1 or 10 Hz, choose file depending on date
    start = time.time()
    if date < dt.datetime(2020, 1, 27):
        file = f"{date:%Y%m%d}_DSHIP_seapath_1Hz.dat"
    else:
        file = f"{date:%Y%m%d}_DSHIP_seapath_10Hz.dat"
    # set encoding and separator, skip the rows with the unit and type of measurement
    seapath = pd.read_csv(f"{path}/{file}", encoding='windows-1252', sep="\t", skiprows=(1, 2),
                          index_col='date time')
    # transform index to datetime
    seapath.index = pd.to_datetime(seapath.index, infer_datetime_format=True)
    seapath.index.name = 'datetime'
    seapath.columns = ['Heading [°]', 'Heave [m]', 'Pitch [°]', 'Roll [°]']  # rename columns
    print(f"Done reading in Seapath data in {time.time() - start:.2f} seconds")
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
    t1 = time.time()
    print("Calculating Heave Rate...")
    # angles in radians
    pitch = np.deg2rad(seapath["Pitch [°]"])
    roll = np.deg2rad(seapath["Roll [°]"])
    yaw = np.deg2rad(seapath["Heading [°]"])
    # time delta between two time steps in seconds
    d_t = np.ediff1d(seapath.index).astype('float64') / 1e9
    if not use_cross_product:
        print("using a simple geometric approach")
        if not only_heave:
            print("using also the roll and pitch induced heave")
            pitch_heave = x_radar * np.tan(pitch)
            roll_heave = y_radar * np.tan(roll)

        elif only_heave:
            print("using only the ships heave")
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
        print("using the cross product approach from Hannes Griesche")
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
            print("Transform into Earth Coordinate System")
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

    print(f"Done with heave rate calculation in {time.time() - t1:.2f} seconds")
    return seapath


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


def calc_heave_corr(container, date, seapath, mean_hr=True):
    """Calculate heave correction for mean Doppler velocity

    Args:
        container (larda container): LIMRAD94 C1/2/3_Range, SeqIntTime, ts, MaxVel, DoppLen
        date (dt.datetime): date of file
        seapath (pd.DataFrame): Data frame with heave rate column ("Heave Rate [m/s]")
        mean_hr (bool): whether to use the mean heave rate over the SeqIntTime or the heave rate at the start time of the chirp

    Returns: heave_corr
        heave_corr (ndarray): heave rate closest to each radar timestep for each height bin, time x range

    """
    start = time.time()
    ####################################################################################################################
    # Calculating Timestamps for each chirp
    ####################################################################################################################
    # timestamp in radar file corresponds to end of chirp sequence with an accuracy of 0.1s
    # make lookup table for chirp durations for each chirptable (see projekt1/remsens/hardware/LIMRAD94/chirptables)
    chirp_durations = pd.DataFrame({"Chirp_No": (1, 2, 3), "tradewindCU": (1.022, 0.947, 0.966),
                                    "Doppler1s": (0.239, 0.342, 0.480), "Cu_small_Tint": (0.225, 0.135, 0.181),
                                    "Cu_small_Tint2": (0.563, 0.573, 0.453)})
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
    chirp_timestamps = pd.DataFrame()
    chirp_timestamps["chirp_1"] = container["ts"] - chirp_dur[0] - chirp_dur[1] - chirp_dur[2]
    chirp_timestamps["chirp_2"] = container["ts"] - chirp_dur[1] - chirp_dur[2]
    chirp_timestamps["chirp_3"] = container["ts"] - chirp_dur[2]

    # array with range bin numbers of chirp borders
    no_chirps = len(chirp_dur)
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
                means_ls.append(seapath[ts_id_diff_min])

        # concatenate all means into one dataframe with the original header (transpose)
        seapath_closest = pd.concat(means_ls, axis=1).T
        # add index with closest seapath time step to radar time step
        seapath_closest.index = seapath.index[id_diff_mins]

        # check if heave rate is greater than 5 standard deviations away from the daily mean and filter those values
        # by averaging the step before and after
        std = np.nanstd(seapath_closest["Heave Rate [m/s]"])
        # try to get indices from values which do not pass the filter. If that doesn't work, then there are no values
        # which don't pass the filter and a ValueError is raised. Write this to a logger
        try:
            id_max = np.asarray(np.abs(seapath_closest["Heave Rate [m/s]"]) > 5 * std).nonzero()[0]
            for j in range(len(id_max)):
                idc = id_max[j]
                warnings.warn(f"Heave rate greater 5 * std encountered ({seapath_closest['Heave Rate [m/s]'][idc]})! \n"
                              f"Using average of step before and after. Index: {idc}", UserWarning)
                # TODO: make more sensible filter, this is a rather sensible filter, because we average over the time
                #  steps before and after. Altough the values are already averages, this should smooth out outliers
                avg_hrate = (seapath_closest["Heave Rate [m/s]"][idc - 1] + seapath_closest["Heave Rate [m/s]"][idc + 1]) / 2
                if avg_hrate > 5 * std:
                    warnings.warn(f"Heave Rate value greater than 5 * std encountered ({avg_hrate})! \n"
                                  f"Even after averaging step before and after too high value! Index: {idc}",
                                  UserWarning)
                seapath_closest["Heave Rate [m/s]"][idc] = avg_hrate
        except ValueError:
            logging.info(f"All heave rate values are within 5 standard deviation of the daily mean!")

        # add column with chirp number to distinguish in quality control
        seapath_closest["Chirp_no"] = np.repeat(i + 1, len(seapath_closest.index))
        # make data frame with used heave rates
        seapath_out = seapath_out.append(seapath_closest)
        # create array with same dimensions as velocity (time, range)
        heave_rate = np.expand_dims(seapath_closest["Heave Rate [m/s]"].values, axis=1)
        # duplicate the heave correction over the range dimension to add it to all range bins of the chirp
        shape = range_bins[i + 1] - range_bins[i]
        heave_corr[:, range_bins[i]:range_bins[i+1]] = heave_rate.repeat(shape, axis=1)
        print(f"Calculated heave correction for Chirp {i+1} in {time.time() - t1:.2f} seconds")

    print(f"Done with heave correction calculation in {time.time() - start:.2f} seconds")
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
    print(f"Done with translation of heave corrections to Doppler bins in {time.time() - start:.2f} seconds")
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
    print(f"Starting heave correction for {date:%Y-%m-%d}")
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
        print(f"Done with heave corrections in {time.time() - start:.2f} seconds")
        return new_vel, heave_corr, seapath_out
    except KeyError:
        print(f"No input Velocities found! Cannot correct Doppler Velocity.\n Returning only heave_corr array!")
        print(f"Done with heave correction calculation only in {time.time() - start:.2f} seconds")
        new_vel = ["I'm an empty list!"]  # create an empty list to return the same number of variables
        return new_vel, heave_corr, seapath_out


def heave_correction_spectra(data, date,
                             path_to_seapath="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP",
                             mean_hr=True, only_heave=False, use_cross_product=True, transform_to_earth=True, add=False):
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

    Returns: A number of variables
        new_spectra (ndarray); corrected Doppler velocities, same shape as data["VHSpec"]["var"] or list if no Doppler
        Spectra are given;
        heave_corr (ndarray): heave rate closest to each radar timestep for each height bin, shape = (time x range);
        seapath_out (pd.DataFrame): data frame with all heave information from the closest time steps to the chirps

    """
    ####################################################################################################################
    # Data Read in
    ####################################################################################################################
    start = time.time()
    print(f"Starting heave correction for {date:%Y-%m-%d}")
    seapath = read_seapath(date, path_to_seapath)

    ####################################################################################################################
    # Calculating Heave Rate
    ####################################################################################################################
    seapath = calc_heave_rate(seapath, only_heave=only_heave, use_cross_product=use_cross_product,
                              transform_to_earth=transform_to_earth)

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

        print(f"Done with heave corrections in {time.time() - start:.2f} seconds")
        return new_spectra, heave_corr, n_dopp_bins_shift, seapath_out
    except KeyError:
        print(f"No input spectra found! Cannot shift spectra.\n Returning only heave_corr and n_dopp_bins_shift array!")
        print(f"Done with heave correction calculation only in {time.time() - start:.2f} seconds")
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
        print(f"Calculating sensitivity curve for program {p}")
        begin_dt = begin_dts[p]
        end_dt = end_dts[p]
        t1 = time.time()
        slv[p] = larda.read(system, "SLv", [begin_dt, end_dt], plot_range)
        slh[p] = larda.read(system, "SLh", [begin_dt, end_dt], plot_range)
        print(f"Read in sensitivity limits in {time.time() - t1:.2f} seconds")

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
            print(f"Read in and interpolated DWD rainflag in {time.time() - t1:.2f} seconds")
        else:
            rain_flag = False

    # take mean of sensitivity limit for whole period of operation
    t1 = time.time()
    mean_slv = {}
    mean_slh = {}
    mean_slv_f = {}
    mean_slh_f = {}
    for p in program:
        mean_slv[p] = np.mean(slv[p]['var'], axis=0)
        mean_slh[p] = np.mean(slh[p]['var'], axis=0)
        if rain_flag:
            # rainflag filtered means
            mean_slv_f[p] = np.mean(np.ma.masked_where(rain_flag_dwd_ip[p] == 1, slv[p]['var']), axis=0)
            mean_slh_f[p] = np.mean(np.ma.masked_where(rain_flag_dwd_ip[p] == 1, slh[p]['var']), axis=0)

    print(f"Averaged sensitivity limits for rain filtered and non filtered data in {time.time() - t1:.2f} seconds")

    # get min and max sensitivity limits for whole period of operation
    t1 = time.time()
    min_max_slv = dict.fromkeys(['min', 'max'], {})
    min_max_slh = dict.fromkeys(['min', 'max'], {})
    min_max_slv_f = dict.fromkeys(['min', 'max'], {})
    min_max_slh_f = dict.fromkeys(['min', 'max'], {})
    for p in program:
        min_max_slv['min'][p] = np.min(slv[p]['var'], axis=0)
        min_max_slh['min'][p] = np.min(slh[p]['var'], axis=0)
        min_max_slv['max'][p] = np.max(slv[p]['var'], axis=0)
        min_max_slh['max'][p] = np.max(slh[p]['var'], axis=0)
        if rain_flag:
            # rainflag filtered min max
            min_max_slv_f['min'][p] = np.min(np.ma.masked_where(rain_flag_dwd_ip[p] == 1, slv[p]['var']), axis=0)
            min_max_slh_f['min'][p] = np.min(np.ma.masked_where(rain_flag_dwd_ip[p] == 1, slh[p]['var']), axis=0)
            min_max_slv_f['max'][p] = np.max(np.ma.masked_where(rain_flag_dwd_ip[p] == 1, slv[p]['var']), axis=0)
            min_max_slh_f['max'][p] = np.max(np.ma.masked_where(rain_flag_dwd_ip[p] == 1, slh[p]['var']), axis=0)

    print(f"Calculated min and max sensitivity limits for rain filtered and non filtered data in "
          f"{time.time() - t1:.2f} seconds")
    print(f"Done with calculate_sensitivity_curve in {time.time() - start:.2f} seconds")
    if rain_flag:
        stats = {'mean_slv': mean_slv, 'mean_slv_f': mean_slv_f, 'mean_slh': mean_slh, 'mean_slh_f': mean_slh_f,
                 'min_slv': min_max_slv['min'], 'min_slv_f': min_max_slv_f['min'],
                 'min_slh': min_max_slh['min'], 'min_slh_f': min_max_slh_f['min'],
                 'max_slv': min_max_slv['max'], 'max_slv_f': min_max_slv_f['max'],
                 'max_slh': min_max_slh['max'], 'max_slh_f': min_max_slh_f['max']}
    else:
        stats = {'mean_slv': mean_slv, 'mean_slh': mean_slh,
                 'min_slv': min_max_slv['min'], 'min_slh': min_max_slh['min'],
                 'max_slv': min_max_slv['max'], 'max_slh': min_max_slh['max']}
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

if __name__ == '__main__':
    import sys, time
    import datetime as dt
    sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
    sys.path.append('.')
    import pyLARDA
    import numpy as np

    larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
    begin_dt = dt.datetime(2020, 2, 5, 0, 0, 5)
    end_dt = dt.datetime(2020, 2, 5, 23, 59, 55)
    plot_range = [0, 'max']
    mdv = larda.read("LIMRAD94_cn_input", "Vel", [begin_dt, end_dt], plot_range)
    moments = {"VEL": mdv}
    for var in ['C1Range', 'C2Range', 'C3Range', 'SeqIntTime', 'Inc_ElA', 'DoppLen', 'MaxVel']:
        print('loading variable from LV1 :: ' + var)
        moments.update({var: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'])})
    new_vel, heave_corr, seapath_out = heave_correction(moments, begin_dt, use_cross_product=True)
    # test without Doppler Velocity input
    moments.__delitem__('VEL')
    new_vel, heave_corr, seapath_out = heave_correction(moments, begin_dt, use_cross_product=True)
    print("Done Testing heave_correction...")
