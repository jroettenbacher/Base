#!/usr/bin/python

"""plot sensitivity curve for each chirp of LIMRAD94 for all chirptables
sensitivity limit is calculated for horizontal and vertical polarization
for more information see LIMRAD94 manual chapter 2.6
filter sensitivity during rain
"""

import sys

# just needed to find pyLARDA from this location
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pyLARDA
import pyLARDA.helpers as h
import datetime as dt
import numpy as np
from scipy import interpolate
import pandas as pd
import logging

log = logging.getLogger('pyLARDA')
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler())

# define plot path
plot_path = "/projekt1/remsens/work/jroettenbacher/plots/sensitivity"
# Load LARDA
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
system = "LIMRAD94"

# read in slv, slh for all 3 chirptables
# read in reflectivity and use mask to filter for rain
# take only non rainy profiles (no signal in 2:6 lowest range gates)
# 3 subplots each for a chirptable with mean sensitivity limit v and h

begin_dts = [dt.datetime(2020, 1, 17, 0, 0, 5), dt.datetime(2020, 1, 30, 15, 30, 5),
             dt.datetime(2020, 1, 31, 22, 30, 5)]
end_dts = [dt.datetime(2020, 1, 27, 0, 0, 5), dt.datetime(2020, 1, 30, 23, 42, 00),
           dt.datetime(2020, 2, 19, 23, 59, 55)]
plot_range = [0, 'max']
# names and program numbers of chirp tables
chirptables = ("tradewindCU (P09)", "Cu_small_Tint (P06)", "Cu_small_Tint2 (P07)")
programs = ["P09", "P06", "P07"]
name = f'{plot_path}/RV-Meteor_LIMRAD94_sensitivity_curves_all_chirptables.png'

# read in sensitivity variables over whole range (all chirps) and reflectivity from cloudnet input
slv = {}
slh = {}
radar_z = {}
rain_flag_dwd = {}
rain_flag_dwd_ip = {}
for begin_dt, end_dt, program in zip(begin_dts, end_dts, programs):
    slv[program] = larda.read(system, "SLv", [begin_dt, end_dt], plot_range)
    slh[program] = larda.read(system, "SLh", [begin_dt, end_dt], plot_range)
    # weather data, time res = 1 min, only read in Dauer (duration) column, gives rain duration in seconds
    weather = pd.read_csv("/projekt2/remsens/data/campaigns/eurec4a/RV-METEOR_DWD/20200114_M161_Nsl.CSV", sep=";",
                          index_col="Timestamp", usecols=[0, 5], squeeze=True)
    weather.index = pd.to_datetime(weather.index, format="%d.%m.%Y %H:%M")
    weather = weather[begin_dt:end_dt]  # select date range
    rain_flag_dwd[program] = weather > 0  # set rain flag if rain duration is greater 0 seconds
    # interpolate rainflag do radar time resolution
    f = interpolate.interp1d(h.dt_to_ts(rain_flag_dwd[program].index), rain_flag_dwd[program], kind='nearest',
                             fill_value="extrapolate")
    rain_flag_dwd_ip[program] = f(np.asarray(slv[program]['ts']))
    # adjust rainflag to sensitivity limit dimensions
    rain_flag_dwd_ip[program] = np.tile(rain_flag_dwd_ip[program], (slv[program]['var'].shape[1], 1)).swapaxes(0, 1).copy()
    # radar_z[program] = larda.read("LIMRAD94_cn_input", "Ze", [begin_dt, end_dt], plot_range)

# take mean of rainflag filtered sensitivity limit
mean_slv = {}
mean_slh = {}
mean_slv_f = {}
mean_slh_f = {}
for program in programs:
    mean_slv[program] = np.mean(slv[program]['var'], axis=0)
    mean_slh[program] = np.mean(slh[program]['var'], axis=0)
    mean_slv_f[program] = np.mean(np.ma.masked_where(rain_flag_dwd_ip[program] == 1, slv[program]['var']), axis=0)
    mean_slh_f[program] = np.mean(np.ma.masked_where(rain_flag_dwd_ip[program] == 1, slh[program]['var']), axis=0)

########################################################################################################################
# PLOTTING
########################################################################################################################
plt.style.use('default')
plt.rcParams.update({'figure.figsize': (16, 9)})

# plot with all chirps and both polarizations
fig, axs = plt.subplots(ncols=3, constrained_layout=True, sharey='all', sharex='all')
for ax, program, i in zip(axs, programs, range(len(chirptables))):
    ax.plot(h.lin2z(mean_slv[program]), slv[program]['rg'], label='vertical')
    ax.plot(h.lin2z(mean_slh[program]), slh[program]['rg'], label='horizontal')
    ax.set_title(chirptables[i])
    ax.set_ylabel("Height [m]")
    ax.set_xlabel("Sensitivity Limit [dBZ]")
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.grid(True, which='both', axis='both', color="grey", linestyle='-', linewidth=1)
    ax.legend(title="Polarization")
fig.suptitle(f"Unfiltered Mean Sensitivity limit for LIMRAD94 \n"
             f"Eurec4a - whole duration of chirptable use", fontsize=16)
fig_name = name.replace(".png", "_unfiltered.png")
fig.savefig(f'{fig_name}', dpi=250)
print(f'figure saved :: {fig_name}')
plt.close()

fig, axs = plt.subplots(ncols=3, constrained_layout=True, sharey='all', sharex='all')
for ax, program, i in zip(axs, programs, range(len(chirptables))):
    ax.plot(h.lin2z(mean_slv_f[program]), slv[program]['rg'], label='vertical')
    ax.plot(h.lin2z(mean_slh_f[program]), slh[program]['rg'], label='horizontal')
    ax.set_title(chirptables[i])
    ax.set_ylabel("Height [m]")
    ax.set_xlabel("Sensitivity Limit [dBZ]")
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.grid(True, which='both', axis='both', color="grey", linestyle='-', linewidth=1)
    ax.legend(title="Polarization")
fig.suptitle(f"Rain Filtered Mean Sensitivity limit for LIMRAD94 \n"
             f"Eurec4a - whole duration of chirptable use", fontsize=16)
fig_name = name.replace(".png", "_filtered.png")
fig.savefig(f'{fig_name}', dpi=250)
print(f'figure saved :: {fig_name}')
plt.close()

# plot the difference between filtered and unfiltered data
fig, axs = plt.subplots(ncols=3, constrained_layout=True, sharey='all', sharex='all')
for ax, program, i in zip(axs, programs, range(len(chirptables))):
    ax.plot(h.lin2z(mean_slv_f[program]) - h.lin2z(mean_slv[program]), slv[program]['rg'], label='vertical')
    ax.plot(h.lin2z(mean_slh_f[program]) - h.lin2z(mean_slh[program]), slh[program]['rg'], label='horizontal')
    ax.set_title(chirptables[i])
    ax.set_ylabel("Height [m]")
    ax.set_xlabel("Sensitivity Limit [dBZ]")
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.grid(True, which='both', axis='both', color="grey", linestyle='-', linewidth=1)
    ax.legend(title="Polarization")
fig.suptitle(f"Difference Between Rain Filtered and Unfiltered Mean Sensitivity Limit for LIMRAD94 \n"
             f"Eurec4a - whole duration of chirptable use", fontsize=16)
fig_name = name.replace(".png", "_filtered-unfiltered.png")
fig.savefig(f'{fig_name}', dpi=250)
print(f'figure saved :: {fig_name}')
plt.close()