#!/usr/bin/python3

import sys
# just needed to find pyLARDA from this location
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda/')
sys.path.append('.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FormatStrFormatter)
import matplotlib.dates as mdates
import pyLARDA
import pyLARDA.helpers as h
import datetime
import numpy as np
import pandas as pd
import re
import time
from scipy import interpolate
import logging
log = logging.getLogger('pyLARDA')
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler())

# Load LARDA
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)

begin_dt = datetime.datetime(2020, 1, 17, 0, 0, 5)
end_dt = datetime.datetime(2020, 1, 19, 23, 59, 55)
plot_range = [0, 'max']
plot_path = f'/projekt1/remsens/work/jroettenbacher/plots'

# read in MWR variables
print(f"Loading HATPRO data...")
start = time.time()
MWR_lwp = larda.read('HATPRO', "LWP", [begin_dt, end_dt])
MWR_lwp['var_unit'] = 'g m-2'
MWR_lwp['var'] = MWR_lwp['var'] * 1000  # conversion to g/m2
MWR_iwv = larda.read('HATPRO', "IWV", [begin_dt, end_dt])
MWR_hum = larda.read('HATPRO', "ABSH", [begin_dt, end_dt], plot_range)
MWR_temp = larda.read('HATPRO', "T", [begin_dt, end_dt], plot_range)

# print(f"Loading LIMRAD94 data for rainflag...")
# radar_Z = larda.read('LIMRAD94', "Ze", [begin_dt, end_dt], [0, 400])  # load first few range gates
# radar_Z['var_unit'] = 'dBZ'

print("Loading DWD Weather Station data")
# weather data, time res = 1 min, only read in Dauer (duration) column, gives rain duration in seconds
weather = pd.read_csv("/projekt2/remsens/data/campaigns/eurec4a/RV-METEOR_DWD/20200114_M161_Nsl.CSV", sep=";", index_col="Timestamp",
                      usecols=[0, 5], squeeze=True)
weather.index = pd.to_datetime(weather.index, format="%d.%m.%Y %H:%M")
weather = weather[begin_dt:end_dt]  # select date range
rain_flag_dwd = weather > 0  # set rain flag if rain duration is greater 0 seconds
# get a one dimensional array with the indices where rainflag turns from True to False or vice versa
indices = np.asarray(np.where(np.diff(rain_flag_dwd))).flatten()
# get indices where rainflag turns from True to False only -> index where it stops to rain
rain_indices = np.asarray([idx for idx in indices if rain_flag_dwd[idx]])
# from the end of each rain event add 10 minutes of masked values
minutes = 10  # just for readability
for i in rain_indices:
    rain_flag_dwd[i:(i+minutes)] = True

# mask rainy values in data
print("Creating Mask...")
# interpolate DWD time resolution to MWR/Radar time resolution
f = interpolate.interp1d(h.dt_to_ts(rain_flag_dwd.index), rain_flag_dwd, kind='nearest', fill_value="extrapolate")
rain_flag_dwd_ip = f(MWR_iwv['ts'])  # interpolate DWD RR to MWR time values
rain_flag_dwd_ip = rain_flag_dwd_ip == 1

# mask values in container from larda
MWR_iwv['var'] = np.ma.masked_where(rain_flag_dwd_ip, MWR_iwv['var'])
MWR_lwp['var'] = np.ma.masked_where(rain_flag_dwd_ip, MWR_lwp['var'])

# create data frame
df = pd.DataFrame({'lwp': MWR_lwp['var'], 'iwv': MWR_iwv['var']}, index=np.array([h.ts_to_dt(ts) for ts in MWR_lwp['ts']]))
# mask values in data frame, mask needs two have same shape as data frame (2 columns, x rows)
df_masked = df.mask(np.array([rain_flag_dwd_ip, rain_flag_dwd_ip]).T)
# mask iwv values > 65 kg m-2
df_masked["iwv"] = df_masked["iwv"].mask(df_masked["iwv"] > 65)
# mask negative lwp values
df_masked["lwp"] = df_masked["lwp"].mask(df_masked["lwp"] < 0)

########################################################################################################################
# Statistics
########################################################################################################################
# calculate statistics for EUREC4A, 12/24h mean/median/std
print("Calculating Statistics...")
t1 = time.time()

# calculate whole campaign statistics
df_masked.describe()
df_masked.var()
df_masked.median()

# calculate statistics for plotting, save data frames to dictionary
stats = dict()
hours = [24]  # define averaging times
for hour in hours:
    # calculate mean and standard deviation in a rolling window of size hour
    df = df_masked.rolling(f"{hour}H").mean()
    std = df_masked.rolling(f"{hour}H").std()
    std.columns = ["lwp_std", "iwv_std"]  # rename columns of std data frame for merging
    # merge data frames on index and save to dictionary
    stats[f"{hour}h_mean"] = df.merge(std, left_index=True, right_index=True)
    # calculate median, merge with standard deviation and save to dictionary
    df = df_masked.rolling(f"{hour}H").median()
    stats[f"{hour}h_median"] = df.merge(std, left_index=True, right_index=True)

print(f"Done with stats in {time.time() - t1:.2f}")

########################################################################################################################
# Plotting
########################################################################################################################
# time series plotting
# name = f"{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}"
# fig, _ = pyLARDA.Transformations.plot_timeseries(MWR_lwp, rg_converter=True, title=True)
# fig.savefig(plot_path + name + '_MWR_LWP.png', dpi=250)
#
# fig, _ = pyLARDA.Transformations.plot_timeseries(MWR_iwv, rg_converter=True, title=True)
# fig.savefig(plot_path + name + '_MWR_IWV.png', dpi=250)
#
# fig, _ = pyLARDA.Transformations.plot_timeheight(MWR_hum, rg_converter=True, title=True)
# fig.savefig(plot_path + name + '_MWR_abshum.png', dpi=250)
#
# fig, _ = pyLARDA.Transformations.plot_timeheight(MWR_temp, rg_converter=True, title=True)
# fig.savefig(plot_path + name + '_MWR_temp.png', dpi=250)

########################################################################################################################
# plotting statistics
# layout stuff
print("Plotting statistics...")
t1 = time.time()
plt.style.use("default")
plt.rcParams.update({'figure.figsize': (8, 6)})
variables = ["lwp", "iwv"]
ylabels = ["Liquid Water Path [g m-2]", "Integrated Water Vapor [kg m-2]"]
titles = ["Liquid Water Path", "Integrated Water Vapor"]
for i in stats:
    string = i.split(sep="_")  # split data frame name to extract averaging time and statistic used
    stat = string[1].capitalize()
    timestep = string[0]
    for var, ylabel, title in zip(variables, ylabels, titles):
        t2 = time.time()
        fig, ax = plt.subplots()
        ax.errorbar(stats[i].index, stats[i][var], yerr=stats[i][f'{var}_std'], fmt='-b',
                    ecolor='lightgrey', label=f"{var.swapcase()} with 1 standard deviation", linewidth=3)
        ax.legend(title='', loc='upper right')
        if var == "iwv":
            ax.set_ylim([15, 50])
        elif var == "lwp":
            ax.set_ylim([0, 350])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Datetime [UTC]")
        ax.set_title(f"{title} {timestep} Rolling {stat}"
                     f"\nEUREC4A RV-Meteor - {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d} "
                     f"\nMicrowave Radiometer HATPRO Uni Leipzig")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_formatter(FormatStrFormatter('%d'))
        ax.tick_params(which='minor', length=4, labelsize=12)
        fig.autofmt_xdate()
        # plt.tight_layout()
        # ax.grid(True, which='minor', color="grey", linestyle='-', linewidth=1)
        ax.xaxis.grid(True, which='major', color="k", linestyle='-', linewidth=2, alpha=0.5)
        ax.yaxis.grid(True, which='major', color="k", linestyle='-', linewidth=2, alpha=0.5)
        figname = f"{plot_path}/RV-Meteor_microwave-radiometer_{timestep}_rolling_{stat}_{var}_" \
                  f"{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}.png"
        plt.savefig(figname, dpi=250)
        print(f"Figure saved :: {figname}\n"
              f"Time passed: {time.time() - t2:.2f}")
        plt.close()
print(f"Finished plotting statistics in {time.time() - t1:.2f}")
