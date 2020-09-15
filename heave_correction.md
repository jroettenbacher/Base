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

After no success with the mean Doppler velocity, the correction is directly applied to the spectra, which are shifted by a number of bins, corresponding to the heave rate.

To detect a possible time shift between the two signals, a cross correlation is performed between the mean Doppler velocity averaged over height and the heave rate interpolated to the same time resolution. For P07 a shift of **1.9** seconds and for P09 a shift of **1.6** seconds was detected and needs to be corrected for.
The result with shifting the heave rate by 19 time steps don't look good. Thus, the next try is shifting by -19 time steps .

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

see PowerPoint $\rightarrow$ do not look good

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

see PowerPoint

## 3. Step by Step

### 3.1 Calculate heave rate for corresponding day
Three components:
* heave rate
* pitch induced heave
* roll induced heave

### 3.2 Calculate heave correction for each chirp 

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

### 3.3 Check for time shift between radar and ship

```python
def calc_time_shift_limrad_seapath(seapath, version=1, **kwargs):
```

* find a day with continuous measurements of Doppler velocity
* average Doppler velocity over height to retrieve time series of mean Doppler velocity
* interpolate heave rate onto radar time
* interpolate NaN values in both time series
* cross correlate the time series and check for time shift with either version 1 or 2 (see function for details)

### 3.4 Apply heave correction to mean Doppler velocity $\rightarrow$ unsuccessful

* decide whether to add or subtract heave rate
* return new array with corrected mean Doppler velocity

### 3.5 Apply heave correction to Doppler spectra

* shift heave rate by given time steps (eg. 19 or 16)
* calculate Doppler resolution
* translate heave rate into Doppler bins
* shift spectra to left or right

## Code

In a private GitHub Repo from Johannes Röttenbacher. Just ask and I can share it with you.

### Quality control

  see Power Point