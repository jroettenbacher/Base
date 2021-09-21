#!/usr/bin/env python
"""Script for humidity comparison between all measurement instruments on RV Meteor
author: Johannes Röttenbacher
"""

# %% module import
import sys

# just needed to find pyLARDA from this location
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')

import matplotlib
import matplotlib.pyplot as plt
import pyLARDA
import pyLARDA.helpers as h
import datetime
import numpy as np
import logging

log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

larda = pyLARDA.LARDA().connect("eurec4a")

# %% read abs hum from LIMHAT
begin_dt = datetime.datetime(2020, 1, 17, 0, 0, 5)
end_dt = datetime.datetime(2020, 2, 19, 23, 59, 55)
time_slice = [begin_dt, end_dt]
range_slice = [0, 10000]
abs_hum_limhat = larda.read("HATPRO", "ABSH", time_slice, range_slice)
t_limhat = larda.read("HATPRO", "T", time_slice, range_slice)

# %% calculate relative humidity from absolute humidity (kg/m3) and temperature (K) using analytical approximation of Clausius-Clapeyron equation
Rw = 462  # specific gas constant for water vapor (J/kg K)
# vapor pressure e0 (Pa) at T0 (K)
e0 = 611
T0 = 273.15
L = (2500 - 2.42 * (t_limhat['var'] - 273.15)) * 1000  # specific heat for evaporation (J/kg)
es = e0 * np.exp((L / (Rw * T0)) * ((t_limhat['var'] - T0) / t_limhat['var']))  # saturation pressure in Pa
e = abs_hum_limhat['var'] * Rw * t_limhat['var']  # water vapor pressure
rh = e / es
rh = rh * 100  # relative humidity

# %% make larda container with relative humidity
rh_limhat = t_limhat  # copy container
rh_limhat['var'] = rh
rh_limhat['var_unit'] = "%"
rh_limhat['var_lims'] = [0, 100]
rh_limhat['name'] = "RH"

# %% convert absolute humidity from kg to g / m3
abs_hum_limhat['var'] = abs_hum_limhat['var'] * 1000
abs_hum_limhat['var_unit'] = 'g m$^{-3}$'
abs_hum_limhat['var_lims'] = [0.0, 20]

# %% plot abs hum LIMHAT
fig, ax = pyLARDA.Transformations.plot_timeheight2(abs_hum_limhat)
fig.savefig("../plots/eurec4a_humidity_comp/RV-METEOR_LIMHAT_abs-hum_20200117-20200219.png", dpi=100)
plt.close()

# %% plot relative humidity LIMHAT
fig, ax = pyLARDA.Transformations.plot_timeheight2(rh_limhat)
fig.savefig("../plots/eurec4a_humidity_comp/RV-METEOR_LIMHAT_rel-hum_20200117-20200219.png", dpi=100)
plt.close()