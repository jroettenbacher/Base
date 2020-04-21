#!/bin/python

# script to compute PDFs of the attitude angles and the heave of the RV-Meteor, LIMRAD94 and the Arduino during Eurec4a
# for different time resolutions
# new feature: plot time series of heave, pitch and roll data

# import libraries
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import datetime
import os
import glob
import re
import sys
# just needed to find pyLARDA from this location
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda/')
sys.path.append('.')

import pyLARDA
import pyLARDA.helpers as h
import datetime
import logging

log = logging.getLogger('pyLARDA')
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler())

# Load LARDA
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)

# local paths JR
# input_path = "./data"
# output_path = "./plots"

# # Catalpa Paths
# input_path = "/home/remsens/data"
# output_path = "/home/remsens/code/larda3/scripts/plots/PDFs_attitude_heave/"

# Catalpa Paths from JR Laptop
input_path = "/projekt2/remsens/data/campaigns/eurec4a"
output_path = "/projekt1/remsens/work/jroettenbacher/plots/attitude"

# define date range, only works for one day at the time
begin_dt = datetime.datetime(2020, 2, 6, 0, 0, 0)
end_dt = datetime.datetime(2020, 2, 10, 23, 59, 59)

########################################################################################################################
# Data Read In
########################################################################################################################
# RV-Meteor
# list all files in directory, read them in and concat them, then select the date range
all_files = sorted(glob.glob(os.path.join(input_path + "/RV-METEOR_DSHIP/*_10Hz.dat")))
file_list = []
for f in all_files:
    # match anything (.*) and the date group (?P<date>) consisting of 8 digits (\d{8})
    match = re.match(r".*(?P<date>\d{8})", f)
    # convert string to datetime
    date_from_file = datetime.datetime.strptime(match.group('date'), '%Y%m%d')
    if begin_dt <= date_from_file <= end_dt:
        file_list.append(f)
rv_meteor = pd.concat(pd.read_csv(f, encoding='windows-1252', sep="\t", skiprows=(1, 2), index_col='date time')
                      for f in file_list)
rv_meteor.index = pd.to_datetime(rv_meteor.index, infer_datetime_format=True)
rv_meteor.index.name = 'datetime'
rv_meteor.columns = ['Heading [°]', 'Heave [m]', 'Pitch [°]', 'Roll [°]']

# Action Log, read in action log of CTD actions
# file_list = sorted(glob.glob(os.path.join(input_path + r"DSHIP_RV-METEOR/*_RV-Meteor_device_action_log.dat")))
# rv_meteor_action = pd.concat((pd.read_csv(f, encoding='windows-1252', sep="\t")
#                              for f in file_list), join='outer')
# rv_meteor_action = pd.read_csv(f"{input_path}DSHIP_RV-METEOR/{begin_dt:%Y%m%d}_RV-Meteor_device_action_log.dat",
#                                encoding='windows-1252', sep='\t')
# rv_meteor_action["Date/Time (Start)"] = pd.to_datetime(rv_meteor_action["Date/Time (Start)"],
#                                                        infer_datetime_format=True)
# rv_meteor_action = rv_meteor_action.loc[begin_dt:end_dt]

########################################################################################################################
# LIMRAD94

radar_roll = larda.read("LIMRAD94_cn_input", "Inc_El", [begin_dt, end_dt])
radar_pitch = larda.read("LIMRAD94_cn_input", "Inc_ElA", [begin_dt, end_dt])

########################################################################################################################
# Arduino
# read in measurement data
all_files = sorted(glob.glob(os.path.join(input_path + "/ARDUINO/final_daily_files/*_attitude_arduino.csv")))
file_list = []
for f in all_files:
    # match anything (.*) and the date group (?P<date>) consisting of 8 digits (\d{8})
    match = re.match(r"(?P<date>\d{8}).*", f)
    # convert string to datetime
    date_from_file = datetime.datetime.strptime(match.group('date'), '%Y%m%d')
    if begin_dt <= date_from_file <= end_dt:
        file_list.append(f)
arduino = pd.concat(pd.read_csv(f, encoding='windows-1252', sep=",", usecols=[1, 2, 3, 4], index_col='datetime')
                    for f in file_list)
arduino.index = pd.to_datetime(arduino.index, infer_datetime_format=True)
arduino.index.name = "datetime"
# rename columns
arduino.columns = ['Heading [°]', 'Pitch [°]', 'Roll [°]']

########################################################################################################################
# PDF Plotting
########################################################################################################################
# RV-Meteor
variables = ["Heave [m]", "Pitch [°]", "Roll [°]"]
# subsample data by 0.5, 1 and 3 seconds
rv_meteor_500ms = rv_meteor.resample("0.5S").mean()
rv_meteor_1s = rv_meteor.resample("1S").mean()
rv_meteor_3s = rv_meteor.resample("3S").mean()
# plot PDFs for each variable
# plt.style.use('ggplot')
# plt.rcParams.update({'font.size': 16, 'figure.figsize': (16, 9)})
# for var in variables:
#     # plot PDF, 40 bins
#     plt.hist([rv_meteor_3s[f'{var}'], rv_meteor_1s[f'{var}'], rv_meteor_500ms[f'{var}'], rv_meteor[f"{var}"]],
#              bins=40, density=True, histtype='bar', log=True, label=["3s", "1s", "0.5s", "0.1s"])
#     plt.legend(title="Sampling Frequency")
#     plt.xlabel(f"{var}")
#     plt.ylabel("Probability Density")
#     plt.title(f"Probability Density Function of {var[:-4]} Motion - DSHIP\n"
#               f" EUREC4A RV-Meteor {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}")
#     plt.tight_layout()
#     # plt.show()
#     filename = f"{output_path}RV-Meteor_Seapath_{var[:-4]}_PDF_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}_log.png"
#     plt.savefig(filename, dpi=300)
#     plt.close()
#     print(f"Figure saved to {filename}")
########################################################################################################################
# LIMRAD94
variables = [radar_roll, radar_pitch]
names = ["Roll", "Pitch"]
# for var, name in zip(variables, names):
#     # plot PDF, 40 bins
#     plt.hist(var['var'], bins=40, density=True, histtype='bar', log=True, label="Median: 1.88s")
#     plt.legend(title="Sampling Frequency")
#     plt.xlabel(f"{name}")
#     plt.ylabel("Probability Density")
#     plt.title(f"Probability Density Function of {name} Motion - LIMRAD94\n"
#               f" EUREC4A RV-Meteor {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}")
#     plt.tight_layout()
#     # plt.show()
#     filename = f"{output_path}RV-Meteor_LIMRAD94_{name}_PDF_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}_log.png"
#     plt.savefig(filename, dpi=300)
#     plt.close()
#     print(f"Figure saved to {filename}")

# plot time series of roll and pitch

# fig, axs = plt.subplots(2, 1, sharex=True)
# axs[0].plot(radar_roll['var'], label='Roll')
# axs[1].plot(radar_pitch['var'], label='Pitch')
# axs[0].set_ylabel("Roll [°]")
# axs[1].set_ylabel("Pitch [°]")
# axs[1].set_xlabel("Datetime [UTC]")
# title = "LIMRAD94 Roll and Pitch Motion\n EUREC4A RV-Meteor "
# if end_dt.date() > begin_dt.date():
#     axs[0].set_title(f"{title}{begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}")
#     filename = f"{output_path}RV-Meteor_LIMRAD94_roll-pitch_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}.png"
#     fig.savefig(filename, dpi=250)
#     print(f"{filename} saved to {output_path}")
# else:
#     axs[0].set_title(f"{title}{begin_dt:%Y-%m-%d}")
#     filename = f"{output_path}RV-Meteor_LIMRAD94_roll-pitch_{begin_dt:%Y%m%d}.png"
#     fig.savefig(filename, dpi=250)
#     print(f"{filename} saved to {output_path}")
# fig.close()
########################################################################################################################
# ARDUINO attitude sensor
variables = ["Pitch [°]", "Roll [°]"]
# subsample data by 0.5, 1 and 3 seconds
arduino_500ms = arduino.resample("0.5S").mean()
arduino_1s = arduino.resample("1S").mean()
arduino_3s = arduino.resample("3S").mean()
# plot PDFs for each variable
# plt.style.use('ggplot')
# plt.rcParams.update({'font.size': 16, 'figure.figsize': (16, 9)})
# for var in variables:
#     # plot PDF, 40 bins
#     plt.hist([arduino_3s[f'{var}'], arduino_1s[f'{var}'], arduino_500ms[f'{var}'], arduino[f"{var}"]],
#              bins=40, density=True, histtype='bar', log=True, label=["3s", "1s", "0.5s", "0.25s"])
#     plt.legend(title="Sampling Frequency")
#     plt.xlabel(f"{var}")
#     plt.ylabel("Probability Density")
#     plt.title(f"Probability Density Function of {var[:-4]} Motion - Arduino sensor\n"
#               f" EUREC4A RV-Meteor {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}")
#     plt.tight_layout()
#     # plt.show()
#     filename = f"{output_path}RV-Meteor_Arduino_attitude_{var[:-4]}_PDF_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}_log.png"
#     plt.savefig(filename, dpi=300)
#     plt.close()
#     print(f"Figure saved to {filename}")
########################################################################################################################
# Plot attitude angles
########################################################################################################################
# RV-Meteor

# set date range to plot
# begin_dt = datetime.datetime(2020, 1, 31, 0, 0, 0)
# end_dt = datetime.datetime(2020, 2, 2, 23, 59, 59)
# use 3 second averaged data to decrease number of points to plot, drop Heading
rv_meteor_plot = rv_meteor_3s.drop('Heading [°]', axis=1).loc[begin_dt:end_dt]
# subset ctd data
# ctd_plot = rv_meteor_action.loc[begin_dt:end_dt]

# plot style options
plt.style.use('ggplot')
plt.rcParams.update({'figure.figsize': (11/2.54, 3.5/2.54)})
# set date tick locator and formater
hfmt = mdates.DateFormatter('%d/%m/%y %H')

########################################################################################################################
# plot heave, pitch and roll RV-Meteor
# fig, axs = plt.subplots(3, 1, sharex=True)
# axs[0].plot(rv_meteor_plot['Heave [m]'], 'r', label='Heave')
# axs[0].set_title(f"Heave, Pitch and Roll measured by the Seapath\n "
#                  f"EUREC4A - RV-Meteor {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}")
# axs[0].set_ylabel("Heave [m]")
# axs[1].plot(rv_meteor_plot['Pitch [°]'], 'b', label='Pitch')
# axs[1].set_ylabel("Pitch [°]")
# axs[2].plot(rv_meteor_plot['Roll [°]'], 'g', label='Roll')
# axs[2].set_ylabel("Roll [°]")
# axs[2].set_xlabel("Datetime [UTC]")
# axs[2].xaxis.set_major_formatter(hfmt)
# fig.autofmt_xdate()
# # plt.show()
# if end_dt.date() > begin_dt.date():
#     plt.savefig(f"{output_path}/RV-Meteor_seapath_heave-pitch-roll_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}.png", dpi=250)
#     print(f"Saved RV-Meteor_seapath_heave-pitch-roll_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}.png to {output_path}")
# else:
#     plt.savefig(f"{output_path}/RV-Meteor_seapath_heave-pitch-roll_{begin_dt:%Y%m%d}.png", dpi=250)
#     print(f"Saved RV-Meteor_seapath_heave-pitch-roll_{begin_dt:%Y%m%d}.png to {output_path}")

########################################################################################################################
# plot only Roll RV-Meteor

# fig, ax = plt.subplots()
# ax.set_title(f"Roll Motion measured by the Seapath\n "
#              f"EUREC4A - RV-Meteor {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}")
# ax.plot(rv_meteor_plot['Roll [°]'], 'g', label='Roll')
# ax.set_ylabel("Roll [°]")
# ax.set_xlabel("Datetime [UTC]")
# ax.xaxis.set_major_formatter(hfmt)
# # add a black vertical line when a CTD started
# # get y limits
# y_lims = ax.get_ylim()
# ax.vlines(ctd_plot.loc[ctd_plot['Action'] == 'in the water'].index, ymin=y_lims[0], ymax=y_lims[1],
#           color='k', lw=3, label="Start of CTD")
# # add a red vertical line when a CTD ended
# ax.vlines(ctd_plot.loc[ctd_plot['Action'] == 'on deck'].index, ymin=y_lims[0], ymax=y_lims[1],
#           color='r', lw=3, label="End of CTD")
# ax.legend()
# fig.autofmt_xdate()
# # plt.show()
# if end_dt.date() > begin_dt.date():
#     plt.savefig(f"{output_path}/RV-Meteor_seapath_roll_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}.png", dpi=250)
#     print(f"Saved RV-Meteor_seapath_roll_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}.png to {output_path}")
# else:
#     plt.savefig(f"{output_path}/RV-Meteor_seapath_roll_{begin_dt:%Y%m%d}.png", dpi=250)
#     print(f"Saved RV-Meteor_seapath_roll_{begin_dt:%Y%m%d}.png to {output_path}")

########################################################################################################################
# plot all three instrument measurements in one plot
variables = ["Roll [°]", "Pitch [°]"]
for var, radar_var in zip(variables, [radar_roll, radar_pitch]):
    fig, ax = plt.subplots()
    ax.set_title(f"{var[:-4]} Motion \n "
                 f"EUREC4A - RV-Meteor {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}")
    ax.plot(rv_meteor[var], 'g', label='Seapath 10Hz')
    ax.plot(arduino[var], 'b', label='Arduino 4Hz')
    ax.plot(radar_roll['var'], 'r', label='LIMRAD94 0.5Hz')
    ax.set_ylabel(var)
    ax.set_xlabel("Datetime [UTC]")
    ax.xaxis.set_major_formatter(hfmt)
    ax.legend(title='Instrument')
    fig.autofmt_xdate()
    if end_dt.date() > begin_dt.date():
        filename = f"{output_path}/RV-Meteor_{var[:-4]}_comparison_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}.png"
        plt.savefig(filename, dpi=250)
        print(f"Saved {filename} to {output_path}")
    else:
        filename = f"{output_path}/RV-Meteor_{var[:-4]}_comparison_{begin_dt:%Y%m%d}.png"
        plt.savefig(filename, dpi=250)
        print(f"Saved {filename} to {output_path}")

########################################################################################################################
# PDF of all three instruments
# variables = ["Roll [°]", "Pitch [°]"]
# for var, radar_var in zip(variables, [radar_roll, radar_pitch]):
#     # plot PDF, 40 bins
#     plt.hist([rv_meteor[var], arduino[var], radar_var['var']],
#              bins=40, density=True, histtype='bar', log=True, label=["Seapath 10Hz", "Arduino 4Hz", "LIMRAD94 0.5Hz"])
#     plt.legend(title="Instrument")
#     plt.xlabel(var)
#     plt.ylabel("Probability Density")
#     title = f"Probability Density Function of {var[:-6]} Motion\n  EUREC4A RV-Meteor "
#     if end_dt.date() > begin_dt.date():
#         plt.title(f"{title}{begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}")
#         plt.tight_layout()
#         filename = f"{output_path}RV-Meteor_all_{var[:-6]}_PDF_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}_log.png"
#         plt.savefig(filename, dpi=250)
#         print(f"{filename} saved to {output_path}")
#     else:
#         plt.title(f"{title}{begin_dt:%Y-%m-%d}")
#         plt.tight_layout()
#         filename = f"{output_path}RV-Meteor_all_{var[:-6]}_PDF_{begin_dt:%Y%m%d}_log.png"
#         plt.savefig(filename, dpi=250)
#         print(f"{filename} saved to {output_path}")
#     plt.close()
