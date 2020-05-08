# LIMRAD94 - Eurec4a Heave Correction

**Problem:**  
A Doppler cloud radar measures vertical fall velocities of hydrometeors. Due to the up and down movement of the 
RV-Meteor (the so called heave), those fall velocities have a systematic error corresponding to the heave rate.  
The heave rate or heave velocity is heave per second and thus has a unit of m/s. Another component is the roll and 
pitch induced heave. Since the radar was placed off center of the ship, the roll and pitch movements of the ship also 
induce a heave motion on the radar.   
**Note:** LIMRAD94 convention: MDV < 0 -> particle falling towards the radar; MDV > 0 -> particle moving away from radar
 
**Solution:**  
Correct the mean Doppler velocity of each chirp by the heave rate, calculated from measurements of the heave by the 
RV-Meteor. To do this a python function called `heave_correction` is written. Because the correction is applied to each 
chirp, the function is called in LIMRAD94_to_Cloudnet_v2.py which calculates the radar moments from the measured Doppler 
spectra.  
Correction:
   
| MDV [m/s] | Heave Rate [m/s] | Measured MDV [m/s] | Correction |  
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

note: the radar timestamp corresponds to the end of the chirp sequence with 0.1s accuracy  
 
Duration of each chirp in seconds by chirp table:  

| Chirp Table | 1. Chirp duration [s] | 2. Chirp duration [s] | 3. Chirp duration [s] |
| --- | --- | --- | --- |
| tradewindCU (P09) | 1.022 | 0.947 | 0.966 |
| Cu_small_Tint2 (P07) | 0.563 | 0.573 | 0.453 |
 
* calculate timestamp for each chirp
* calculate range bins which define the chirp borders
* calculate the absolute difference between all seapath time steps and each radar time step
* select rows which have the minimum difference to the radar time steps
* add heave rate to mean doppler velocity

### Quality control

![uncorrected MDV](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\heave_correction\plots\quality_control\20200205_0000_20200205_2359_3km_cloudnet_input_MDV_uncorrected.png)   
Uncorrected Doppler Velocity  

![uncorrected MDV zoom](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\heave_correction\plots\quality_control\20200205_0900_20200205_1100_3km_cloudnet_input_MDV_uncorrected.png)  
Zoom in of uncorrected Doppler Velocity

![heave correction](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\heave_correction\plots\quality_control\20200205_0000_20200205_2359_3km_cloudnet_input_heave_correction.png)  
Heave correction as calculated by function  

![corrected MDV](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\heave_correction\plots\quality_control\20200205_0000_20200205_2359_3km_cloudnet_input_MDV_corrected.png)  
Corrected Doppler Velocity 

![corrected MDV zoom](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\heave_correction\plots\quality_control\20200205_0900_20200205_1100_3km_cloudnet_input_MDV_corrected.png)  
Zoom in of corrected Doppler Velocity

![difference between corrected and uncorrected](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\heave_correction\plots\quality_control\20200205_0000_20200205_2359_3km_cloudnet_input_MDV_corrected-measured.png)    
Difference between corrected and uncorrected Doppler Velocity

![difference between corrected and uncorrected zoom](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\heave_correction\plots\quality_control\20200205_0900_20200205_1100_3km_cloudnet_input_MDV_corrected-measured.png)      
Zoom in on difference between corrected and uncorrected Doppler Velocity  

![heave components unnormalized](C:\Users\Johannes\Documents\Studium\Hiwi_Kalesse\heave_correction\plots\quality_control\tmp.png)
TODO:
- plot with all three components of heave, normalized by the combined heave for each chirp
- plot with time series of heave rate for each chirp
