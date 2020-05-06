
from itertools import groupby
import numpy as np
import pandas as pd
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
from dateutil.relativedelta import relativedelta


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


def heave_correction(moments, date):
    """Correct mean Doppler velocity for heave motion of ship (RV-Meteor)

    Args:
        moments: LIMRAD94 moments container as returned by spectra2moments in spec2mom_limrad94.py
        date (datetime.datetime): object with date of current file

    Returns:
        new_vel (ndarray); corrected Doppler velocities, same shape as moments["VEL"]["var"]
        heave_corr (ndarray): heave rate closest to each radar timestep for each height bin, same shape as
        moments["VEL"]["var"]
        seapath_chirptimes (pd.DataFrame): data frame with a column for each Chirp, containing the timestamps of the
        corresponding heave rate
        seapath_out (pd.DataFrame): data frame with all heave information from the closest time steps to the chirps

    """
    # position of radar in relation to Measurement Reference Unit (Seapath) of RV-Meteor in meters
    x_radar = -11
    y_radar = 4.07
    ####################################################################################################################
    # Data Read in
    ####################################################################################################################
    start = time.time()
    print(f"Starting heave correction for {date:%Y-%m-%d}")
    input_path = "/projekt2/remsens/data/campaigns/eurec4a/RV-METEOR_DSHIP"
    ####################################################################################################################
    # Seapath attitude and heave data 1 or 10 Hz, choose file depending on date
    if date < dt.datetime(2020, 1, 27):
        file = f"{date:%Y%m%d}_DSHIP_seapath_1Hz.dat"
    else:
        file = f"{date:%Y%m%d}_DSHIP_seapath_10Hz.dat"
    seapath = pd.read_csv(f"{input_path}/{file}", encoding='windows-1252', sep="\t", skiprows=(1, 2),
                          index_col='date time')
    seapath.index = pd.to_datetime(seapath.index, infer_datetime_format=True)
    seapath.index.name = 'datetime'
    seapath.columns = ['Heading [°]', 'Heave [m]', 'Pitch [°]', 'Roll [°]']
    print(f"Done reading in Seapath data in {time.time() - start:.2f} seconds")

    ####################################################################################################################
    # Calculating Heave Rate
    ####################################################################################################################
    t1 = time.time()
    print("Calculating Heave Rate...")
    # sum up heave, pitch induced and roll induced heave
    pitch = np.deg2rad(seapath["Pitch [°]"])
    roll = np.deg2rad(seapath["Roll [°]"])
    pitch_heave = x_radar * np.tan(pitch)
    roll_heave = y_radar * np.tan(roll)
    seapath["radar_heave"] = seapath["Heave [m]"] + pitch_heave + roll_heave
    # add pitch and roll induced heave to data frame to include in output for quality checking
    seapath["pitch_heave"] = pitch_heave
    seapath["roll_heave"] = roll_heave
    # ediff1d calculates the difference between consecutive elements of an array
    # heave difference / time difference = heave rate
    heave_rate = np.ediff1d(seapath["radar_heave"]) / np.ediff1d(seapath.index).astype('float64') * 1e9
    # the first calculated heave rate corresponds to the second time step
    heave_rate = pd.DataFrame({'Heave Rate [m/s]': heave_rate}, index=seapath.index[1:])
    seapath = seapath.join(heave_rate)
    print(f"Done with heave rate calculation in {time.time() - t1:.2f} seconds")

    ####################################################################################################################
    # Calculating Timestamps for each chirp and add closest heave rate to corresponding Doppler velocity
    ####################################################################################################################
    # timestamp in radar file corresponds to end of chirp sequence with an accuracy of 0.1s
    # make lookup table for chirp durations for each chirptable (see projekt1/remsens/hardware/LIMRAD94/chirptables)
    chirp_durations = pd.DataFrame({"Chirp_No": (1, 2, 3), "tradewindCU": (1.022, 0.947, 0.966),
                                    "Cu_small_Tint2": (0.563, 0.573, 0.453)})
    # calculate end of each chirp by subtracting the duration of the later chirp(s) + half the time of the chirp itself
    # the timestamp then corresponds to the middle of the chirp
    # select chirp durations according to date
    if date < dt.datetime(2020, 2, 1):
        chirp_dur = chirp_durations["tradewindCU"]
    else:
        chirp_dur = chirp_durations["Cu_small_Tint2"]
    chirp_timestamps = pd.DataFrame()
    chirp_timestamps["chirp_1"] = moments['VEL']["ts"] - (chirp_dur[0] / 2) - chirp_dur[1] - chirp_dur[2]
    chirp_timestamps["chirp_2"] = moments['VEL']["ts"] - (chirp_dur[1] / 2) - chirp_dur[2]
    chirp_timestamps["chirp_3"] = moments['VEL']["ts"] - (chirp_dur[2] / 2)

    # create new Doppler velocity by adding the heave rate of the closest time step
    # list with range bin numbers of chirp borders
    no_chirps = len(chirp_dur)
    range_bins = np.zeros(no_chirps + 1, dtype=np.int)  # needs to be length 4 to include all +1 chirp borders
    for i in range(no_chirps):
        try:
            range_bins[i + 1] = range_bins[i] + moments[f'C{i + 1}Range']['var'][0].shape
        except ValueError:
            # in case only one file is read in data["C1Range"]["var"] has only one dimension
            range_bins[i + 1] = range_bins[i] + moments[f'C{i + 1}Range']['var'].shape

    seapath_ts = seapath.index.values.astype(np.float64) / 10 ** 9  # convert datetime index to seconds since 1970-01-01
    # initialize output variables
    new_vel = np.empty_like(moments['VEL']['var'])  # dimensions (time, range)
    heave_corr = np.empty_like(moments['VEL']['var'])
    seapath_chirptimes = pd.DataFrame()
    seapath_out = pd.DataFrame()
    for i in range(no_chirps):
        t1 = time.time()
        # select only velocities from one chirp
        var = moments['VEL']['var'][:, range_bins[i]:range_bins[i+1]]
        # convert timestamps of moments to array
        ts = chirp_timestamps[f"chirp_{i+1}"].values
        id_diff_min = []  # initialize list for indices of the time steps with minumum difference
        for t in ts:
            # calculate the absolute difference between all seapath time steps and the radar time step
            abs_diff = np.abs(seapath_ts - t)
            # minimum difference
            min_diff = np.min(abs_diff)
            # find index of minimum difference
            # use argmax to return only the first index where condition is true
            id_diff_min.append(np.argmax(abs_diff == min_diff))
        # select the rows which are closest to the radar time steps
        seapath_closest = seapath.iloc[id_diff_min].copy()
        # add column with chirp number to distinguish in quality control
        seapath_closest["Chirp_no"] = np.repeat(i + 1, len(seapath_closest.index))
        # create array with same dimensions as velocity (time, range)
        heave_rate = np.expand_dims(seapath_closest["Heave Rate [m/s]"].values, axis=1)
        # duplicate the heave correction over the range dimension to add it to all range bins of the chirp
        heave_corr[:, range_bins[i]:range_bins[i+1]] = heave_rate.repeat(var.shape[1], axis=1)
        new_vel[:, range_bins[i]:range_bins[i+1]] = var + heave_corr[:, range_bins[i]:range_bins[i+1]]
        # save chirptimes of seapath for quality control, as seconds since 1970-01-01 00:00 UTC
        seapath_chirptimes[f"Chirp_{i+1}"] = seapath_closest.index.values.astype(np.int64) / 10 ** 9
        seapath_out = seapath_out.append(seapath_closest, ignore_index=True)
        print(f"Corrected Doppler velocities in Chirp {i+1} in {time.time() - t1:.2f} seconds")

    return new_vel, heave_corr, seapath_chirptimes, seapath_out


if __name__ == '__main__':
    import sys, time
    import datetime as dt
    sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
    sys.path.append('.')
    import pyLARDA
    import pyLARDA.helpers as h
    import logging
    import numpy as np

    larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
    begin_dt = dt.datetime(2020, 1, 17, 0, 0, 5)
    end_dt = dt.datetime(2020, 1, 17, 23, 59, 55)
    plot_range = [0, 'max']
    mdv = larda.read("LIMRAD94_cn_input", "Vel", [begin_dt, end_dt], plot_range)
    moments = {"VEL": mdv}
    for var in ['MaxVel', 'DoppLen', 'C1Range', 'C2Range', 'C3Range']:
        print('loading variable from LV1 :: ' + var)
        moments.update({var: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'])})
    new_vel, heave_corr, seapath_chirptimes, seapath_out = heave_correction(moments, begin_dt)
    print("Done Testing heave_correction...")
