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
import pyLARDA.Transformations as pltrans
import datetime
import numpy as np
import logging

log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

larda = pyLARDA.LARDA().connect("eurec4a")

plot_path = "../plots/eurec4a_humidity_comp"

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
abs_hum_limhat['var_lims'] = [0.0, 30]

# %% plot abs hum LIMHAT
fig, ax = pltrans.plot_timeheight2(abs_hum_limhat)
fig.savefig(f"{plot_path}/RV-METEOR_LIMHAT_abs-hum_20200117-20200219.png", dpi=100)
plt.close()

# %% plot relative humidity LIMHAT
fig, ax = pltrans.plot_timeheight2(rh_limhat)
fig.savefig(f"{plot_path}/RV-METEOR_LIMHAT_rel-hum_20200117-20200219.png", dpi=100)
plt.close()

# %% retrieve certain height ranges from abs hum; 60 (50), 120 (100), 300 (325) and 400 m
abs_hum_limhat_50 = pltrans.slice_container(abs_hum_limhat, value={"range": [50]})
abs_hum_limhat_100 = pltrans.slice_container(abs_hum_limhat, value={"range": [100]})
abs_hum_limhat_325 = pltrans.slice_container(abs_hum_limhat, value={"range": [325]})
abs_hum_limhat_400 = pltrans.slice_container(abs_hum_limhat, value={"range": [400]})
# abs_hum_limhat["rg"]  # possible height bins
rh_limhat_50 = pltrans.slice_container(rh_limhat, value={"range": [50]})
rh_limhat_100 = pltrans.slice_container(rh_limhat, value={"range": [100]})
rh_limhat_325 = pltrans.slice_container(rh_limhat, value={"range": [325]})
rh_limhat_400 = pltrans.slice_container(rh_limhat, value={"range": [400]})

# %% plot abs hum time series of certain height bins in one plot
CB_color_cycle = ["#6699CC", "#117733", "#CC6677", "#DDCC77"]  # colorblind friendly color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=CB_color_cycle)
fig, ax = plt.subplots(figsize=[14, 5.7])
for plot_data, height in zip([abs_hum_limhat_50, abs_hum_limhat_100, abs_hum_limhat_325, abs_hum_limhat_400],
                             [50, 100, 325, 400]):
    fig, ax = pltrans.plot_timeseries2(plot_data, figure=fig, axis=ax, label=f"{height}m",
                                       title="RV-Meteor LIMHAT Absolute Humidity")
    print(f"Done with {height}m")

ax.grid()
ax.legend()
fig.savefig(f"{plot_path}/RV-METEOR_LIMHAT_abs-hum_timeseries_20200117-20200219.png", dpi=100)
plt.close()

# %% plot abs hum time series of certain height bins in one plot each
for plot_data, height in zip([abs_hum_limhat_50, abs_hum_limhat_100, abs_hum_limhat_325, abs_hum_limhat_400],
                             [50, 100, 325, 400]):
    fig, ax = pltrans.plot_timeseries2(plot_data, label=f"{height}m", title="RV-Meteor LIMHAT Absolute Humidity")
    ax.grid()
    ax.legend()
    fig.savefig(f"{plot_path}/RV-METEOR_LIMHAT_abs-hum_{height}m_20200117-20200219.png", dpi=100)
    plt.close()
    print(f"Done with {height}m")

# %% plot rel hum time series of certain height bins in one plot
fig, ax = plt.subplots(figsize=[14, 5.7])

for plot_data, height in zip([rh_limhat_50, rh_limhat_100, rh_limhat_325, rh_limhat_400], [50, 100, 325, 400]):
    fig, ax = pltrans.plot_timeseries2(plot_data, figure=fig, axis=ax, label=f"{height}m",
                                       title="RV-Meteor LIMHAT Relative Humidity")
    print(f"Done with {height}m")

ax.grid()
ax.legend()
fig.savefig(f"{plot_path}/RV-METEOR_LIMHAT_rel-hum_timeseries_20200117-20200219.png", dpi=100)
plt.close()

# %% plot rel hum time series of certain height bins in one plot each
for plot_data, height in zip([rh_limhat_50, rh_limhat_100, rh_limhat_325, rh_limhat_400], [50, 100, 325, 400]):
    fig, ax = pltrans.plot_timeseries2(plot_data, label=f"{height}m", title="RV-Meteor LIMHAT Relative Humidity")
    ax.grid()
    ax.legend()
    fig.savefig(f"{plot_path}/RV-METEOR_LIMHAT_rel-hum_{height}m_20200117-20200219.png", dpi=100)
    plt.close()
    print(f"Done with {height}m")

