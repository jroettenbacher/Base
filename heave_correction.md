# LIMRAD94 - Eurec4a Heave Correction

1. [Introduction](#introduction)
2. [Different ways to calculate heave rate](#different-ways-to-calculate-heave-rate)
3. [Step by Step](#step-by-step)

## 1. Introduction 

For any questions please contact: johannes.roettenbacher@web.de

**Problem:**  
A Doppler cloud radar measures vertical fall velocities of hydrometeors. Due to the up and down movement of the RV-Meteor (the so called heave), those fall velocities have a systematic error corresponding to the heave rate.  
The heave rate or heave velocity is heave per second and thus has an unit of m/s. Another component is the roll and pitch induced heave. Since the radar was placed off center of the ship, the roll and pitch movements of the ship also induce a heave motion onto the radar.  
**Note:** LIMRAD94 convention: MDV < 0 $\rightarrow$ particle falling towards the radar; MDV > 0 $\rightarrow$ particle moving away from radar

**Solution:**  
Correct the mean Doppler velocity of each chirp by the heave rate, calculated from measurements of the motion angles by the RV-Meteor. 

**Progress description:**  
In the following the development of the whole correction is described. 

*Calculate heave rate*

The first step is to calculate the heave rate at the radar position. For this an approach from Hannes Griesche was applied (see Sec. 2.3). Claudia Acquistapage from the Uni Cologne, who was also working on motion correction for the same radar, developed a different approach (see Sec. 2.4). The difference between the two can be seen in Sec. 2.5. the difference between the two is probably related to the fact, that Claudia also shifts the ship time stamps by half the sample frequency, so that they correspond with the center of the measurement.

*Time shift correction*

Another problem that had to be considered was a possible time shift between the radar and the ship time. Although both times were retrieved by a GPS sensor, there was a possibility that the signal processing of the radar would take some time before the time stamp was written to the measurement. To detect a possible time shift between the two signals, a cross correlation was performed between the mean Doppler velocity averaged over height and the heave rate interpolated to the same time resolution. For P07 a shift of **1.9** seconds and for P09 a shift of **1.6** seconds was detected and needed to be corrected for.
Radar time lacks behind the ship time, therefore the ship time was shifted back in time.

Claudia uses a more precise approach by calculating the time shift for every hour and every chirp, when possible. Thus, this approach was adapted in is now the one which is used.

*Correcting for heave rate*

At first the whole correction was only applied to the mean Doppler velocity as reported by the radar. To do this a python function called `heave_correction` was  written. By using the chirp duration times for each chirp, exact time stamps for each chirp could be calculated, allowing for the possibility to correct the heave rate in each chirp. This is possible because the Measurement Reference Unit (MRU) of the RV-Meteor measured at higher sampling frequency as the radar. 

It was quickly discovered, that it would make more sense to directly correct the Doppler spectra, which are shifted by a number of bins, corresponding to the heave rate. The corresponding function: `heave_correction_spectra`

After the heave correction the Doppler spectra are also dealiased.

**Correction:** 

| Real MDV [m/s] <br />measured in Radar CS | Heave Rate [m/s] <br />measured in Ship CS | Measured MDV [m/s] | Corrected MDV [m/s] |
| ----------------------------------------- | ------------------------------------------ | ------------------ | ------------------- |
| +3, particle moving up                    | +1, ship moving down                       | +3 + (+)1 = +4     | +4 - (+)1 = 3       |
| +3                                        | -1, ship moving up                         | +3 + (-)1 = +2     | +2 - (-)1 = 3       |
| -3, particle moving down                  | +1, ship moving down                       | -3 + (+1) = -2     | -2 - (+)1 = -3      |
| -3                                        | -1, ship moving up                         | -3 + (-)1 = -4     | -4 - (-)1 = -3      |

*CS: Coordinate System*

**Problem:** The results look better if the heave rate is subtracted (spectra moved to the left)!
**Probable Solution:** The ship and earth coordinate system are defined with the z-axis downward, meaning positive is down. Whereas the radar coordinate system in which the MDV is measured is defined with the z-axis upward, meaning positive is up. Thus, to correct the measured Doppler velocity the sign has to be switched.

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

see PowerPoint $\rightarrow$ do not look good
To simplistic!

### 2.3 Using the cross product 

**Idea:** Determine the heave rate $v_{C_z}$ of the radar by summing up the z-component of the cross product between the rotation vector of the ship $v_{P_R}$ and the position of the radar relative to the INS of the ship $X_R$ with the z-component of the translation vector of the ship $v_{P_{T,z}}$. That's the way Hannes Griesche did it ([paper](https://doi.org/10.5194/amt-2019-434)).

This determines the cross product in the ships coordinate system. The cross product needs to be transformed into the earths coordinate system.

$\implies$ transform the cross product $v_R$ to earth coordinates with help of a transformation matrix as shown in [Hill 2005](https://repository.library.noaa.gov/view/noaa/17400).

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

see PowerPoint $\rightarrow$ this approach can be used

### 2.4 Transforming the position vector to the earth coordinate system

**Idea:** Transform the position vector of the radar $$X_R$$ into the earth coordinate system for each time step by using the transformation matrix. Extract the heave rate by dividing the change in the z-direction of $X_R$ by the time resolution of the ship measurements.

**What you need:**

* Displacement of radar in reference to Inertial Navigation System
* Roll, pitch and yaw from INS

### 2.5 Difference between 2.3 and 2.4

<img src="C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\heave_correction\heave_rate_diff_jr-ca.png" alt="heave_rate_diff_jr-ca" style="zoom:24%;" />

## 3. Step by Step

The main program, which can be run via the command line, is called `create_limrad_calibrated_eurec4a.py`. It reads in the Doppler spectra from the LV0 hourly nc files via pyLARDA. For this it uses the function `load_spectra_rpgfmcw94()` from `SpectraProcessing.py`. In this script all other following functions can also be found, with detailed explanations regarding their arguments. The following gives an overview, which steps are covered by which functions. They are all called within the function described in Sec. 3.5.

### 3.1 Calculate heave rate for corresponding day

#### 3.1.1 Cross Product Approach

```python
def calc_heave_rate(seapath, only_heave=False, use_cross_product=True, transform_to_earth=True)
```
Three components:
* heave
* pitch induced heave
* roll induced heave

**Options**:

* only_heave (bool): use only the heave and neglect pitch and roll induced heave
* use_cross_product (bool): use the cross product approach as mentioned above
* transform_to_earth (bool): transform cross product into earth coordinates

#### 3.1.2 Transform to Earth Approach

```python
def calc_heave_rate_claudia(data, x_radar=-11, y_radar=4.07, z_radar=-15.8)
```

### 3.2 Calculate heave correction matrix

#### 3.2.1 Johannes' approach

```python
def calc_heave_corr(container, date, seapath, mean_hr=True)
```

**Note**: the radar timestamp corresponds to the end of the chirp sequence with 0.1s accuracy  

Duration of each chirp in seconds by chirp table:  

| Chirp Table | 1. Chirp duration [s] | 2. Chirp duration [s] | 3. Chirp duration [s] |
| --- | --- | --- | --- |
| tradewindCU (P09) | 1.022 | 0.947 | 0.966 |
| Doppler1s (P02)      | 0.239                 | 0.342                 | 0.480                 |
| Cu_small_Tint (P06) | 0.225 | 0.135 | 0.181 |
| Cu_small_Tint2 (P07) | 0.563 | 0.573 | 0.453 |

* calculate timestamp for each chirp $\rightarrow$ done by larda
  * subtract chirp duration(s) from timestamp to get start time of each chirp
* calculate range bins which define the chirp borders
* find the closest seapath time step to each chirp time step
* take the mean of the heave rate over the integration time of each chirp (SeqIntTime in radar nc file)
* filter heave rates greater 5 standard deviations away from the daily mean and replace by average of the mean heave rate before and after
* create an array with the same dimension as the mean Doppler velocity

**Options**:

* mean_hr (bool): use the mean heave rate over the integration time of each chirp

#### 3.2.2 Claudia's Approach

```python
def calc_corr_matrix_claudia(radar_ts, radar_rg, rg_borders_id, chirp_ts_shifted, Cs_w_radar)
```

interpolates the heave rate to the shifted chirp time stamp

### 3.3 Check for time shift between radar and ship

#### 3.3.1 Johannes' approach

```python
def calc_time_shift_limrad_seapath(seapath, version=1, **kwargs):
```

* find a day with continuous measurements of Doppler velocity
* average Doppler velocity over height to retrieve time series of mean Doppler velocity
* interpolate heave rate onto radar time
* interpolate NaN values in both time series
* cross correlate the time series and check for time shift with either version 1 or 2 (see function for details)

#### 3.3.2 Claudia's approach

```python
def calc_shifted_chirp_timestamps(radar_ts, radar_mdv, chirp_ts, rg_borders_id, n_ts_run, Cs_w_radar, **kwargs)
```

Calculate the exact radar time stamp for each hour and each chirp including the calculated time shift.
Calls the following functions:

```python
def calc_time_shift(w_radar_meanCol, delta_t_min, delta_t_max, resolution, w_ship_chirp, timeSerieRadar, pathFig, chirp, hour, date)
```

**Input:**

* continuous time series of mean Doppler velocity averaged over one chirp (from: `find_mdv_series`)
* heave rate corresponding to that chirp
* time stamps of MDV time series

**Processing:**

* interpolate heave rate onto time stamps of mean Doppler velocity series
* shift heave rate time series by a small delta T
* compute covariance between both time series
* find time shift at which covariance is maximal

```python
def find_mdv_time_series(mdv_values, radar_time, n_ts_run)
```

Given a 2D array of MDV, finds a time series with the minimum amount of nan values and returns an average over height of that series.

### 3.4 Apply heave correction to mean Doppler velocity $\rightarrow$ unsuccessful

```python
def heave_correction(moments, date, path_to_seapath="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP", mean_hr=True, only_heave=False, use_cross_product=True, transform_to_earth=True, add=False)
```

* decide whether to add or subtract heave rate
* return new array with corrected mean Doppler velocity

### 3.5 Apply heave correction to Doppler spectra

```python
def heave_correction_spectra(data, date, path_to_seapath="/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP", mean_hr=True, only_heave=False, use_cross_product=True, transform_to_earth=True, add=False, **kwargs)
```

* calls all of the above functions
* shift heave rate by given time steps (eg. 19 or 16) (deprecated)
* calculate Doppler resolution
* translate heave rate into Doppler bins
* shift spectra to left or right

## Code

Part of [pyLARDA](https://github.com/lacros-tropos/larda).