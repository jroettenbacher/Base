# LIMRAD94 - Eurec4a Heave Correction

**Problem:**  
A Doppler cloud radar measures vertical fall velocities of hydrometeors. Due to the up and down movement of the RV-Meteor (the so called heave), those fall velocities have a systematic error corresponding to the heave rate.  
The heave rate or heave velocity is heave per second and thus has a unit of m/s. Another component is the roll and pitch induced heave. Since the radar was placed off center of the ship, the roll and pitch movements of the ship also induce a heave motion on the radar.  
**Note:** LIMRAD94 convention: MDV < 0 -> particle falling towards the radar; MDV > 0 -> particle moving away from radar

**Solution:**  
Correct the mean Doppler velocity of each chirp by the heave rate, calculated from measurements of the heave by the RV-Meteor. To do this a python function called `heave_correction` is written. Because the correction is applied to each chirp, the function is called in LIMRAD94_to_Cloudnet_v2.py which calculates the radar moments from the measured Doppler spectra.  

**Correction:** 

| Real MDV [m/s] | Heave Rate [m/s] | Measured MDV [m/s] | Corrected MDV [m/s] |
| --- | --- | --- | --- |
| +3 | +1 | +3 - (+)1 = +2 | +2 + (+)1 = 3 |
| +3 | -1 | +3 - (-)1 = +4 | +4 + (-)1 = 3 |
| -3 | +1 | -3 - (+1) = -4 | -4 + (+)1 = -3 |
| -3 | -1 | -3 - (-)1 = -2 | -2 + (-)1 = -3 |

## Step by Step

### 1. Calculate heave rate for corresponding day
Three components:
* heave rate
* pitch induced heave
* roll induced heave

### 2. Correct Each Chirp's Mean Doppler Velocity

**Note**: the radar timestamp corresponds to the end of the chirp sequence with 0.1s accuracy  

Duration of each chirp in seconds by chirp table:  

| Chirp Table | 1. Chirp duration [s] | 2. Chirp duration [s] | 3. Chirp duration [s] |
| --- | --- | --- | --- |
| tradewindCU (P09) | 1.022 | 0.947 | 0.966 |
| Doppler1s (P02)      | 0.239                 | 0.342                 | 0.480                 |
| Cu_small_Tint (P06) | 0.225 | 0.135 | 0.181 |
| Cu_small_Tint2 (P07) | 0.563 | 0.573 | 0.453 |

* calculate timestamp for each chirp
* calculate range bins which define the chirp borders
* calculate the absolute difference between all seapath time steps and each radar time step
* select rows which have the minimum difference to the radar time steps
* filter heave rates greater +- 5 m/s and replace by average of time step before and after
* add heave rate to mean doppler velocity

## Code

```python
def heave_correction(moments, date, path_to_seapath="/projekt2/remsens/data/campaigns/eurec4a/RV-METEOR_DSHIP",
                     only_heave=False):
    """Correct mean Doppler velocity for heave motion of ship (RV-Meteor)

    Args:
        moments: LIMRAD94 moments container as returned by spectra2moments in spec2mom_limrad94.py, C1/2/3_Range
        date (datetime.datetime): object with date of current file
        path_to_seapath (string): path where seapath measurement files (daily dat files) are stored
        only_heave (bool): whether to use only heave to calculate the heave rate or include pitch and roll induced heave

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
    ####################################################################################################################
    # Seapath attitude and heave data 1 or 10 Hz, choose file depending on date
    if date < dt.datetime(2020, 1, 27):
        file = f"{date:%Y%m%d}_DSHIP_seapath_1Hz.dat"
    else:
        file = f"{date:%Y%m%d}_DSHIP_seapath_10Hz.dat"
    seapath = pd.read_csv(f"{path_to_seapath}/{file}", encoding='windows-1252', sep="\t", skiprows=(1, 2),
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
    if not only_heave:
        pitch_heave = x_radar * np.tan(pitch)
        roll_heave = y_radar * np.tan(roll)
    else:
        pitch_heave = 0
        roll_heave = 0
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
                                    "Doppler1s": (0.239, 0.342, 0.480), "Cu_small_Tint": (0.225, 0.135, 0.181),
                                    "Cu_small_Tint2": (0.563, 0.573, 0.453)})
    # calculate end of each chirp by subtracting the duration of the later chirp(s) + half the time of the chirp itself
    # the timestamp then corresponds to the middle of the chirp
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
    chirp_timestamps["chirp_1"] = moments['VEL']["ts"] - (chirp_dur[0] / 2) - chirp_dur[1] - chirp_dur[2]
    chirp_timestamps["chirp_2"] = moments['VEL']["ts"] - (chirp_dur[1] / 2) - chirp_dur[2]
    chirp_timestamps["chirp_3"] = moments['VEL']["ts"] - (chirp_dur[2] / 2)

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
        id_diff_min = []  # initialize list for indices of the time steps with minimum difference
        # TODO: Parallelize this for loop if possible, this takes the most time
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

        # check if heave rate is greater than 5 m/s and filter those values by averaging the step before and after
        id_max = np.asarray(np.abs(seapath_closest["Heave Rate [m/s]"]) > 5).nonzero()[0]
        for j in range(len(id_max)):
            idc = id_max[j]
            avg_hrate = (seapath_closest["Heave Rate [m/s]"][idc - 1] + seapath_closest["Heave Rate [m/s]"][idc + 1]) / 2
            seapath_closest["Heave Rate [m/s]"][idc] = avg_hrate

        # add column with chirp number to distinguish in quality control
        seapath_closest["Chirp_no"] = np.repeat(i + 1, len(seapath_closest.index))
        # create array with same dimensions as velocity (time, range)
        heave_rate = np.expand_dims(seapath_closest["Heave Rate [m/s]"].values, axis=1)
        # duplicate the heave correction over the range dimension to add it to all range bins of the chirp
        heave_corr[:, range_bins[i]:range_bins[i+1]] = heave_rate.repeat(var.shape[1], axis=1)
        # create new Doppler velocity by adding the heave rate of the closest time step
        new_vel[:, range_bins[i]:range_bins[i+1]] = var + heave_corr[:, range_bins[i]:range_bins[i+1]]
        # save chirptimes of seapath for quality control, as seconds since 1970-01-01 00:00 UTC
        seapath_chirptimes[f"Chirp_{i+1}"] = seapath_closest.index.values.astype(np.float64) / 10 ** 9
        # make data frame with used heave rates
        seapath_out = seapath_out.append(seapath_closest)
        print(f"Corrected Doppler velocities in Chirp {i+1} in {time.time() - t1:.2f} seconds")

    # set masked values back to -999 because they also get corrected
    new_vel[moments['VEL']['mask']] = -999
    print(f"Done with heave corrections in {time.time() - start:.2f} seconds")
    return new_vel, heave_corr, seapath_chirptimes, seapath_out
```



### Quality control

| Uncorrected Doppler Velocity                                 | Corrected Doppler Velocity                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="quality_control\20200205_0000_20200205_2359_3km_cloudnet_input_MDV_uncorrected.png" alt="uncorrected MDV"  /> | <img src="quality_control\20200205_0000_20200205_2359_3km_cloudnet_input_MDV_corrected.png" alt="corrected MDV"  /> |
| <img src="quality_control\20200205_0900_20200205_1100_3km_cloudnet_input_MDV_uncorrected.png" alt="uncorrected MDV zoom"  /> | <img src="quality_control\20200205_0900_20200205_1100_3km_cloudnet_input_MDV_corrected.png" alt="corrected MDV zoom"  /> |

  

| heave correction                                             | corrected - uncorrected                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![heave correction](quality_control\20200205_0000_20200205_2359_3km_cloudnet_input_heave_correction.png) | ![difference between corrected and uncorrected](quality_control\20200205_0000_20200205_2359_3km_cloudnet_input_MDV_corrected-measured.png) |
| ![20200205_0900_20200205_1100_3km_cloudnet_input_heave_correction](quality_control\20200205_0900_20200205_1100_3km_cloudnet_input_heave_correction.png) | ![difference between corrected and uncorrected zoom](quality_control\20200205_0900_20200205_1100_3km_cloudnet_input_MDV_corrected-measured.png) |



![heave components unnormalized](C:\Users\Johannes\PycharmProjects\Base\quality_control\heave_elements.png)  
Heave components unnormalized

![heave_rate_by_chirp](C:\Users\Johannes\PycharmProjects\Base\quality_control\heave_rate_by_chirp.png)

Heave Rate by chirp