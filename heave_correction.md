# LIMRAD94 - Eurec4a Heave Correction

**Problem:**  
A Doppler cloud radar measures vertical fall velocities of hydrometeors. Due to the up and down movement of the 
RV-Meteor (the so called heave), those fall velocities have a systematic error corresponding to the heave rate.  
The heave rate or heave velocity is heave per second and thus has a unit of m/s. Another component is the roll and 
pitch induced heave. Since the radar was placed off center of the ship, the roll and pitch movements of the ship also 
induce a heave motion on the radar. 
 
**Solution:**  
Correct the mean Doppler velocity of each chirp by the heave rate, calculated from measurements of the heave by the 
RV-Meteor. To do this a python function called `heave_correction` is written. Because the correction is applied to each 
chirp, the function is called in LIMRAD94_to_Cloudnet_v2.py which calculates the radar moments from the measured Doppler 
spectra.  

## Step by Step

### 1. Calculate heave rate for corresponding day
Three components:
* heave rate
* pitch induced heave
* roll induced heave

### 2. Correct Each Chirp's Mean Doppler Velocity

note: the radar timestamp corresponds to the end of the chirp sequence  
Duration of each chirp in seconds by chirp table:  

| Chirp Table | 1. Chirp duration | 2. Chirp duration | 3. Chirp duration |
| --- | --- | --- | --- |
| tradewindCU (P09) | 1.022 | 0.947 | 0.966 |
| Cu_small_Tint2 (P07) | 0.563 | 0.573 | 0.453 |
 
* calculate timestamp for each chirp
*  