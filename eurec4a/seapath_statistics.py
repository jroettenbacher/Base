#!/bin/python
"""script to compute PDFs of the attitude angles of the Seapath during Eurec4a for different time resolutions
Input: dship data (1Hz and 10Hz)
Output: pdf plots"""

# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import datetime
import os
import glob
import re
import functions_jr as jr

# local paths
input_path = "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/RV-METEOR_DSHIP"
output_path = "/projekt1/remsens/work/jroettenbacher/plots/eurec4a_seapath_attitude"
input_radar = "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/LIMRAD94/angles"
# set options
save_fig = True

# define date range
# if you choose dates before 27.01.2020 Seapath resolution was only 1Hz, adjust filename, expect errors when trying
# to read in files from both resolutions
begin_dt = datetime.datetime(2020, 1, 17, 0, 0, 0)
end_dt = datetime.datetime(2020, 2, 28, 23, 59, 59)

########################################################################################################################
# Data Read In
########################################################################################################################
# RV-Meteor - Seapath
# list all files in directory, select the date range, then read them in and concat them
t1 = time.time()
all_files = sorted(glob.glob(os.path.join(input_path + "/*seapath_*Hz.dat")))
file_list = []
for f in all_files:
    # match anything (.*) and the date group (?P<date>) consisting of 8 digits (\d{8})
    match = re.match(r".*(?P<date>\d{8})", f)
    # convert string to datetime
    date_from_file = datetime.datetime.strptime(match.group('date'), '%Y%m%d')
    if begin_dt <= date_from_file <= end_dt:
        file_list.append(f)
seapath = pd.concat(pd.read_csv(f, encoding='windows-1252', sep="\t", skiprows=(1, 2), index_col='date time',
                                na_values=-999) for f in file_list)
seapath.index = pd.to_datetime(seapath.index, infer_datetime_format=True)
seapath.index.name = 'datetime'
seapath.columns = ['Heading [°]', 'Heave [m]', 'Pitch [°]', 'Roll [°]']
print(f"Done with seapath data read in {time.time() - t1:.2f} seconds")

# LIMRAD94 Attitude angles
t1 = time.time()
all_files = sorted(glob.glob(os.path.join(input_radar + "/RV-Meteor_cloudradar_attitude-angles_*.csv")))
file_list = []
for f in all_files:
    # match anything (.*) and the date group (?P<date>) consisting of 8 digits (\d{8})
    match = re.match(r".*(?P<date>\d{8})", f)
    # convert string to datetime
    date_from_file = datetime.datetime.strptime(match.group('date'), '%Y%m%d')
    if begin_dt <= date_from_file <= end_dt:
        file_list.append(f)
radar = pd.concat(pd.read_csv(f, sep=",", index_col='datetime', parse_dates=True) for f in file_list)
radar.columns = ["unix_time", "Roll [°]", "Pitch [°]"]
print(f"Done with radar data read in {time.time() - t1:.2f} seconds")

########################################################################################################################
# calculate roll and pitch induced heave
########################################################################################################################
t1 = time.time()
# position of radar in relation to Measurement Reference Unit (Seapath) of RV-Meteor in meters
x_radar = -11
y_radar = 4.07
pitch = np.deg2rad(seapath["Pitch [°]"])
roll = np.deg2rad(seapath["Roll [°]"])
pitch_heave = x_radar * np.tan(pitch)
roll_heave = y_radar * np.tan(roll)

# sum all heaves to radar heave
seapath["Radar Heave [m]"] = seapath["Heave [m]"] + pitch_heave + roll_heave

# calculate heave rate
seapath = jr.calc_heave_rate(seapath)
print(f"Done with heave rate calculation in {time.time() - t1:.2f} seconds")

########################################################################################################################
# PDF Plotting
########################################################################################################################
# RV-Meteor Heave, Pitch and Roll
variables = ["Heave [m]", "Pitch [°]", "Roll [°]", "Radar Heave [m]"]
# subsample data by 0.5, 1 and 3 seconds
seapath_500ms = seapath.resample("0.5S").mean()
seapath_1s = seapath.resample("1S").mean()
seapath_3s = seapath.resample("3S").mean()
# plot PDFs for each variable
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16, 'figure.figsize': (16, 9)})
for var in variables:
    # plot PDF, 40 bins
    plt.hist([seapath_3s[f'{var}'], seapath_1s[f'{var}'], seapath_500ms[f'{var}'], seapath[f"{var}"]],
             bins=40, density=True, histtype='bar', log=True, label=["3s", "1s", "0.5s", "0.1s"])
    plt.legend(title="Sampling Frequency")
    plt.xlabel(f"{var}")
    plt.ylabel("Probability Density")
    plt.title(f"Probability Density Function of {var[:-4]} Motion - DSHIP\n"
              f" EUREC4A RV-Meteor {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}")
    plt.tight_layout()
    if save_fig:
        filename = f"{output_path}/RV-Meteor_Seapath_{var[:-4].replace(' ', '_')}_PDF_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}_log.png"
        plt.savefig(filename, dpi=300)
        print(f"Figure saved to {filename}")
    else:
        plt.savefig(f"./tmp/{var}.png")
    plt.close()

########################################################################################################################
# Heave Rate

# RV-Meteor heave rate at radar position
# plot PDF, 40 bins
plt.hist([seapath["Heave Rate [m/s]"]],
         bins=40, density=True, histtype='bar', log=True, label=["Seapath Heave Rate"])
plt.legend()
plt.xlabel(f"Heave Rate [m/s]")
plt.ylabel("Probability Density")
plt.title(f"Probability Density Function of Heave Rate - DSHIP\n"
          f" EUREC4A RV-Meteor {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}")
plt.tight_layout()
if save_fig:
    filename = f"{output_path}/RV-Meteor_Seapath_Heave_Rate_PDF_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}_log.png"
    plt.savefig(filename, dpi=300)
    print(f"Figure saved to {filename}")
else:
    plt.savefig("./tmp/heaverate.png")
plt.close()

########################################################################################################################
# Roll and Pitch from both instruments
# subsample data by 3 seconds to level out gaps in radar data
radar_3s = radar.resample("3S").mean()
# plot PDFs for each variable
variables = ["Roll [°]", "Pitch [°]"]
for var in variables:
    # plot PDF, 40 bins
    plt.hist([seapath_3s[f'{var}'], radar_3s[f'{var}']],
             bins=40, density=True, histtype='bar', log=True, label=["Seapath 3s", "Radar 3s"])
    plt.legend(title="Instrument")
    plt.xlabel(f"{var}")
    plt.ylabel("Probability Density")
    plt.title(f"Probability Density Function of {var[:-4]} Motion - DSHIP\n"
              f" EUREC4A RV-Meteor {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}")
    plt.tight_layout()
    if save_fig:
        filename = f"{output_path}/RV-Meteor_Seapath-Radar_{var[:-4].replace(' ', '_')}_PDF_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}_log.png"
        plt.savefig(filename, dpi=300)
        print(f"Figure saved to {filename}")
    else:
        plt.savefig(f"./tmp/{var}_comp.png")
    plt.close()
