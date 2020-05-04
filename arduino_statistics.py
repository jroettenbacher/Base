#!/bin/python

# script to compute PDFs of the attitude angles of the Arduino during Eurec4a
# for different time resolutions

# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import glob
import re

# local paths
input_path = "./data"
output_path = "./plots"

# define date range, only works for one day at the time
begin_dt = datetime.datetime(2020, 2, 6, 0, 0, 0)
end_dt = datetime.datetime(2020, 2, 10, 23, 59, 59)

########################################################################################################################
# Data Read In
########################################################################################################################
# Arduino
# read in measurement data
all_files = sorted(glob.glob(os.path.join(input_path + "/ARDUINO/final_daily_files/*_attitude_arduino.csv")))
file_list = []
for f in all_files:
    # match anything (.*) and the date group (?P<date>) consisting of 8 digits (\d{8})
    match = re.match(r".*(?P<date>\d{8}).*", f)
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
# ARDUINO attitude sensor
variables = ["Pitch [°]", "Roll [°]"]
# subsample data by 0.5, 1 and 3 seconds
arduino_500ms = arduino.resample("0.5S").mean()
arduino_1s = arduino.resample("1S").mean()
arduino_3s = arduino.resample("3S").mean()
# plot PDFs for each variable
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16, 'figure.figsize': (16, 9)})
for var in variables:
    # plot PDF, 40 bins
    plt.hist([arduino_3s[f'{var}'], arduino_1s[f'{var}'], arduino_500ms[f'{var}'], arduino[f"{var}"]],
             bins=40, density=True, histtype='bar', log=True, label=["3s", "1s", "0.5s", "0.25s"])
    plt.legend(title="Sampling Frequency")
    plt.xlabel(f"{var}")
    plt.ylabel("Probability Density")
    plt.title(f"Probability Density Function of {var[:-4]} Motion - Arduino sensor\n"
              f" EUREC4A RV-Meteor {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}")
    plt.tight_layout()
    # plt.show()
    filename = f"{output_path}RV-Meteor_Arduino_attitude_{var[:-4]}_PDF_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}_log.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Figure saved to {filename}")
