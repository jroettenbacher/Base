#!/bin/python

########################################################################################################################
# library import
########################################################################################################################
import time
import datetime as dt
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def heave_correction(moments, date):
    """Correct mean Doppler velocity for heave motion of ship (RV-Meteor)

    Args:
        moments: LIMRAD94 moments container as returned by spectra2moments in spec2mom_limrad94.py
        date: datetime object with date of current file

    Returns:
        new_vel: ndarray with corrected Doppler velocities, same shape as moments["VEL"]["var"]
        seapath_chirptimes: pandas DataFrame with a column for each Chirp,
                            containing the timestamps of the corresponding heave rate

    """
    # position of radar in relation to Measurement Reference Unit (Seapath) of RV-Meteor in meters
    x_radar = -11
    y_radar = 4.07
    ####################################################################################################################
    # Data Read in
    ####################################################################################################################
    start = time.time()
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
    print(f"Done reading in Seapath data in {time.time() - start}")

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
    # ediff1d calculates the difference between consecutive elements of an array
    # heave difference / time difference = heave rate
    heave_rate = np.ediff1d(seapath["radar_heave"]) / np.ediff1d(seapath.index).astype('float64') * 1e9
    # the first calculated heave rate corresponds to the second time step
    heave_rate = pd.DataFrame({'Heave Rate [m/s]': heave_rate}, index=seapath.index[1:])
    seapath = seapath.join(heave_rate)
    print(f"Done with heave rate calculation in {time.time() - t1}")

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
    range_bins = 0
    new_vel = np.empty_like(moments['VEL']['var'])
    seapath_chirptimes = pd.DataFrame()
    for i in range(len(chirp_dur)):
        range_bins = range_bins + moments[f'C{i+1}Range']['var'].shape[1]  # get number of range bins
        var = moments['VEL']['var'][:, :range_bins]  # select only velocities from one chirp
        # convert timestamps of moments to datetime objects
        ts = pd.Series([dt.datetime.utcfromtimestamp(ts) for ts in chirp_timestamps[f"chirp_{i+1}"].values])
        # calculate the absolute difference between all seapath time steps and each radar time step
        abs_diff = [np.abs(seapath.index - t) for t in ts]
        # select the rows which are closest to the radar time steps
        seapath_closest = seapath.iloc[np.where(abs_diff == np.min(abs_diff))]
        # create array with same dimensions as velocity (time, range)
        heave_correction = np.expand_dims(seapath_closest["Heave Rate [m/s]"].values, axis=1)
        # duplicate the heave correction over the range dimension to add it to all range bins
        new_vel[:, :range_bins] = var + heave_correction.repeat(var.shape[1], axis=1)
        # save chirptimes of seapath for quality control, as seconds since 1970-01-01 00:00 UTC
        seapath_chirptimes[f"Chirp_{i+1}"] = seapath_closest.index.values.astype(np.int64) / 10 ** 9

    return new_vel, seapath_chirptimes
