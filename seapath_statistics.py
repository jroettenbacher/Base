#!/bin/python

# script to compute PDFs of the attitude angles of the Seapath during Eurec4a
# for different time resolutions

# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import datetime
import os
import glob
import re

# local paths
input_path = "/projekt2/remsens/data/campaigns/eurec4a/RV-METEOR_DSHIP"
output_path = "/projekt1/remsens/work/jroettenbacher/plots/heave_correction"

# define date range
# if you choose dates before 27.01.2020 Seapath resolution was only 1Hz, adjust filename, expect errors when trying
# to read in files from both resolutions
begin_dt = datetime.datetime(2020, 1, 27, 0, 0, 0)
end_dt = datetime.datetime(2020, 2, 19, 23, 59, 59)

########################################################################################################################
# Data Read In
########################################################################################################################
# RV-Meteor - Seapath
# list all files in directory, select the date range, then read them in and concat them
t1 = time.time()
all_files = sorted(glob.glob(os.path.join(input_path + "/*_10Hz.dat")))
file_list = []
for f in all_files:
    # match anything (.*) and the date group (?P<date>) consisting of 8 digits (\d{8})
    match = re.match(r".*(?P<date>\d{8})", f)
    # convert string to datetime
    date_from_file = datetime.datetime.strptime(match.group('date'), '%Y%m%d')
    if begin_dt <= date_from_file <= end_dt:
        file_list.append(f)
seapath = pd.concat(pd.read_csv(f, encoding='windows-1252', sep="\t", skiprows=(1, 2), index_col='date time',
                                na_values='-999-999.-999-999-999') for f in file_list)
seapath.index = pd.to_datetime(seapath.index, infer_datetime_format=True)
seapath.index.name = 'datetime'
seapath.columns = ['Heading [°]', 'Heave [m]', 'Pitch [°]', 'Roll [°]']
print(f"Done with data read in {time.time() - t1:.2f} seconds")

# calculate roll and pitch induced heave
# position of radar in relation to Measurement Reference Unit (Seapath) of RV-Meteor in meters
t1 = time.time()
x_radar = -11
y_radar = 4.07
pitch = np.deg2rad(seapath["Pitch [°]"])
roll = np.deg2rad(seapath["Roll [°]"])
pitch_heave = x_radar * np.tan(pitch)
roll_heave = y_radar * np.tan(roll)

# sum all heaves to radar heave
seapath["Radar Heave [m]"] = seapath["Heave [m]"] + pitch_heave + roll_heave

# calculate heave rate
heave_rate = np.ediff1d(seapath["Heave [m]"]) / np.ediff1d(seapath.index).astype('float64') * 1e9
heave_rate = pd.DataFrame({'Heave Rate [m/s]': heave_rate}, index=seapath.index[1:])
seapath = seapath.join(heave_rate)
heave_rate = np.ediff1d(seapath["Radar Heave [m]"]) / np.ediff1d(seapath.index).astype('float64') * 1e9
heave_rate = pd.DataFrame({'Radar Heave Rate [m/s]': heave_rate}, index=seapath.index[1:])
seapath = seapath.join(heave_rate)
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
    # plt.show()
    filename = f"{output_path}/RV-Meteor_Seapath_{var[:-4].replace(' ', '_')}_PDF_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}_log.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Figure saved to {filename}")

########################################################################################################################
# Heave Rate

# RV-Meteor heave and Radar Heave
# plot PDF, 40 bins
plt.hist([seapath["Heave Rate [m/s]"], seapath["Radar Heave Rate [m/s]"]],
         bins=40, density=True, histtype='bar', log=True, label=["Seapath Heave Rate", "Radar Heave Rate"])
plt.legend(title="Sampling Frequency")
plt.xlabel(f"Heave Rate [m/s]")
plt.ylabel("Probability Density")
plt.title(f"Probability Density Function of Heave Rate - DSHIP\n"
          f" EUREC4A RV-Meteor {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}")
plt.tight_layout()
# plt.show()
filename = f"{output_path}/RV-Meteor_Seapath_Heave_Rate_PDF_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}_log.png"
plt.savefig(filename, dpi=300)
plt.close()
print(f"Figure saved to {filename}")
