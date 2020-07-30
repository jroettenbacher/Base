# LIMRAD94 - Eurec4a Heave Correction

1. [Introduction](#introduction)
2. [Different ways to calculate heave rate](#different-ways-to-calculate-heave-rate)
3. [Step by Step](#step-by-step)

## 1. Introduction 

**Problem:**  
A Doppler cloud radar measures vertical fall velocities of hydrometeors. Due to the up and down movement of the RV-Meteor (the so called heave), those fall velocities have a systematic error corresponding to the heave rate.  
The heave rate or heave velocity is heave per second and thus has an unit of m/s. Another component is the roll and pitch induced heave. Since the radar was placed off center of the ship, the roll and pitch movements of the ship also induce a heave motion onto the radar.  
**Note:** LIMRAD94 convention: MDV < 0 $\rightarrow$ particle falling towards the radar; MDV > 0 $\rightarrow$ particle moving away from radar

**Solution:**  
Correct the mean Doppler velocity of each chirp by the heave rate, calculated from measurements of the heave by the RV-Meteor. To do this a python function called `heave_correction` is written. By using the chirp table times for each chirp, the correction can be applied to each chirp.  

**Correction:** 

| Real MDV [m/s] <br />measured in Radar CS | Heave Rate [m/s] <br />measured in Ship CS | Measured MDV [m/s] | Corrected MDV [m/s] |
| ----------------------------------------- | ------------------------------------------ | ------------------ | ------------------- |
| +3, particle moving up                    | +1, ship moving down                       | +3 + (+)1 = +4     | +4 - (+)1 = 3       |
| +3                                        | -1, ship moving up                         | +3 + (-)1 = +2     | +2 - (-)1 = 3       |
| -3, particle moving down                  | +1, ship moving down                       | -3 + (+1) = -2     | -2 - (+)1 = -3      |
| -3                                        | -1, ship moving up                         | -3 + (-)1 = -4     | -4 - (-)1 = -3      |

*CS: Coordinate System*

**Problem:** The results look better if the heave rate is subtracted!
**Probable Solution:** The ship and earth coordinate system are defined with the z-axis downward, meaning positive is down. Whereas the radar coordinate system in which the MDV is measured is defined with the z-axis upward, meaning positive is up. Thus to correct the measured Doppler velocity the sign has to be switched.

## 2. Different Ways to Calculate Heave Rate

### 2.1 Placement of Radar on Ship

![image-20200730092641679](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\heave_correction\documents\pictures\image-20200730092641679.png)

(sketch by Heike Kalesse)

### 2.2 Simple geometric way

**Idea**: Sum up all three heave components for each time step, which are heave from the ship, heave induced by the roll  and heave induced by the pitch of the ship. Divide the change in heave by the time difference between two measurements.

![image-20200730092730949](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\heave_correction\documents\pictures\pitch_induced_heave.png)
Pitch induced heave (Heike Kalesse)

![image-20200730092827389](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\heave_correction\documents\pictures\roll_induced_heave.png)
Roll induced heave (Heike Kalesse)

**What you need:** 

* Displacement of radar in reference to Inertial Navigation System
* Roll, Pitch and Heave from INS

**Math**

$$heave_{radar} = heave_{ship} + x_{radar} * \tan(pitch_{ship}) + y_{radar} * \tan(roll_{ship})$$

$$ heaverate_{radar} = \frac{heave_{radar}(t_{n+1}) - heave_{radar}(t_n)}{t_{n+1} - t_n} $$

**Results**

see powerpoint

### 2.3 Using the cross product 

**Idea:** Determine the heave rate $v_{C_z}$ of the radar by summing up the z-component of the cross product between the rotation vector of the ship $v_{P_R}$ and the position of the radar relative to the INS of the ship $X_R$ with the z-component of the translation vector of the ship $v_{P_{T,z}}$. That's the way Hannes Griesche did it ([paper](https://doi.org/10.5194/amt-2019-434)).

This determines the cross product in the ships coordinate system. The cross product needs to be transformed into the earths coordinate system.

To transform the cross product $v_R$ to earth coordinates with help of a transformation matrix as shown in [Hill 2005](https://repository.library.noaa.gov/view/noaa/17400).

**What you need:**

* Displacement of radar in reference to Inertial Navigation System
* Roll rate, pitch rate and heave rate from INS
* Roll, pitch and yaw from INS

**Math**

-*Cross Product*-

![cross_product_Hannes](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\heave_correction\documents\cross_product_Hannes.png)

(Griesche et al. 2020)

$v_{R_z} = P_{pitch} * y_R - P_{roll} * x_R$

-*Transformation to earth coordinate system*-

![image-20200730093837974](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\heave_correction\documents\pictures\transformation_matrix.png)

(Hill 2005)

$$v_{R}^{E} = Q^T v_R$$
with $v_{R}^{E}$ being the cross product in earth coordinates.

-*Summation*-

$v_{C_z} = v_{R_z}^{E} + v_{P_{T,z}}$

**Results**

see powerpoint

## 3. Step by Step

### 1. Calculate heave rate for corresponding day
Three components:
* heave rate
* pitch induced heave
* roll induced heave

### 2. Calculate heave correction for each chirp 

**Note**: the radar timestamp corresponds to the end of the chirp sequence with 0.1s accuracy  

Duration of each chirp in seconds by chirp table:  

| Chirp Table | 1. Chirp duration [s] | 2. Chirp duration [s] | 3. Chirp duration [s] |
| --- | --- | --- | --- |
| tradewindCU (P09) | 1.022 | 0.947 | 0.966 |
| Doppler1s (P02)      | 0.239                 | 0.342                 | 0.480                 |
| Cu_small_Tint (P06) | 0.225 | 0.135 | 0.181 |
| Cu_small_Tint2 (P07) | 0.563 | 0.573 | 0.453 |

* calculate timestamp for each chirp
  * subtract chirp duration(s) from timestamp to get start time of each chirp
* calculate range bins which define the chirp borders
* find the closest seapath time step to each chirp time step
* take the mean of the heave rate over the integration time of each chirp (SeqIntTime in radar nc file)
* filter heave rates greater 5 standard deviations away from the daily mean and replace by average of the mean heave rate before and after
* create an array with the same dimension as the mean Doppler velocity

### 3. Apply heave correction to mean Doppler velocity

* decide whether to add or subtract heave rate
* return new array with corrected mean Doppler velocity

## Code

```python
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


def calc_heave_corr(container, date, seapath):
    """Calculate heave correction for mean Doppler velocity

    Args:
        container (larda container): LIMRAD94 C1/2/3_Range, SeqIntTime and ts
        date (dt.datetime): date of file
        seapath (pd.DataFrame): Data frame with heave rate column ("Heave Rate [m/s]")

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

    # list with range bin numbers of chirp borders
    no_chirps = len(chirp_dur)
    range_bins = np.zeros(no_chirps + 1, dtype=np.int)  # needs to be length 4 to include all +1 chirp borders
    for i in range(no_chirps):
        try:
            range_bins[i + 1] = range_bins[i] + container[f'C{i + 1}Range']['var'][0].shape
        except ValueError:
            # in case only one file is read in data["C1Range"]["var"] has only one dimension
            range_bins[i + 1] = range_bins[i] + container[f'C{i + 1}Range']['var'].shape

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
        dfs = []  # initialize list for means over integration time for each radar time step
        for t in ts:
            id_diff_min = h.argnearest(seapath_ts, t)  # find index of nearest seapath time step to radar time step
            id_diff_mins.append(id_diff_min)
            # get time stamp of closest index
            ts_diff = seapath.index[id_diff_min]
            # select rows from closest time stamp to end of integration time and average, append to list
            dfs.append(seapath[ts_diff:ts_diff+int_time].mean())

        # concatinate all means into one dataframe with the original header (transpose)
        seapath_closest = pd.concat(dfs, axis=1).T
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


def heave_correction(moments, date, path_to_seapath="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP",
                     only_heave=False, use_cross_product=True, transform_to_earth=True, add=False):
    """Correct mean Doppler velocity for heave motion of ship (RV-Meteor)
    Calculate heave rate from seapath measurements and create heave correction array. If Doppler velocity is given as an
    input, correct it and return an array with the corrected Doppler velocities.
    Without Doppler Velocity input, only the heave correction array is returned.

    Args:
        moments: LIMRAD94 moments container as returned by spectra2moments in spec2mom_limrad94.py, C1/2/3_Range,
                 SeqIntTime and Inc_ElA (for time (ts)) from LV1 file
        date (datetime.datetime): object with date of current file
        path_to_seapath (string): path where seapath measurement files (daily dat files) are stored
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
    heave_corr, seapath_out = calc_heave_corr(container, date, seapath)

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
```



### Quality control

  see Power Point