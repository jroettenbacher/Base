#!/bin/python

# script to compute PDFs of the attitude angles of the Seapath during Eurec4a
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

# define date range
# if you choose dates before 27.01.2020 Seapath resolution was only 1Hz, adjust filename, expect errors when trying
# to read in files from both resolutions
begin_dt = datetime.datetime(2020, 2, 6, 0, 0, 0)
end_dt = datetime.datetime(2020, 2, 10, 23, 59, 59)

########################################################################################################################
# Data Read In
########################################################################################################################
# RV-Meteor - Seapath
# list all files in directory, select the date range, then read them in and concat them
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
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16, 'figure.figsize': (16, 9)})
for var in variables:
    # plot PDF, 40 bins
    plt.hist([rv_meteor_3s[f'{var}'], rv_meteor_1s[f'{var}'], rv_meteor_500ms[f'{var}'], rv_meteor[f"{var}"]],
             bins=40, density=True, histtype='bar', log=True, label=["3s", "1s", "0.5s", "0.1s"])
    plt.legend(title="Sampling Frequency")
    plt.xlabel(f"{var}")
    plt.ylabel("Probability Density")
    plt.title(f"Probability Density Function of {var[:-4]} Motion - DSHIP\n"
              f" EUREC4A RV-Meteor {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}")
    plt.tight_layout()
    # plt.show()
    filename = f"{output_path}RV-Meteor_Seapath_{var[:-4]}_PDF_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}_log.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Figure saved to {filename}")
