#!/bin/python

# Script to calculate heave rate from DSHIP data and plot it

########################################################################################################################
# library import
########################################################################################################################
import time
import datetime as dt
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

########################################################################################################################
# Data Read in
########################################################################################################################
start = time.time()
input_path = "/projekt2/remsens/data/campaigns/eurec4a/RV-METEOR_DSHIP"
plot_path = "/projekt1/remsens/work/jroettenbacher/plots/heave_correction"
# begin and end date
begin_dt = dt.datetime(2020, 1, 17)
end_dt = dt.datetime(2020, 1, 17)
########################################################################################################################
# Seapath attitude and heave data 10 Hz
# list all files in directory that match *_10Hz.dat
all_files = sorted(glob.glob(f"{input_path}/*_10Hz.dat"))
file_list = []  # initiate list
# extract date from filename and check if it lies between begin_dt and end_dt
print(f"Reading in 10Hz Seapath data for dates: {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}...")
for f in all_files:
    # match anything (.*) and the date group (?P<date>) consisting of 8 digits (\d{8})
    match = re.match(r".*(?P<date>\d{8})", f)
    # convert string to datetime
    date_from_file = dt.datetime.strptime(match.group('date'), '%Y%m%d')
    if begin_dt <= date_from_file <= end_dt:
        file_list.append(f)
rv_meteor = pd.concat(pd.read_csv(f, encoding='windows-1252', sep="\t", skiprows=(1, 2), index_col='date time')
                      for f in file_list)
rv_meteor.index = pd.to_datetime(rv_meteor.index, infer_datetime_format=True)
rv_meteor.index.name = 'datetime'
rv_meteor.columns = ['Heading [°]', 'Heave [m]', 'Pitch [°]', 'Roll [°]']
print(f"Done reading in Seapath data in {time.time() - start}")

########################################################################################################################
# Calculating Heave Rate
########################################################################################################################
t1 = time.time()
print("Calculating Heave Rate...")
heave_rate = np.ediff1d(rv_meteor["Heave [m]"]) / np.ediff1d(rv_meteor.index).astype('float64') * 1e9
heave_rate = pd.DataFrame({'Heave Rate [m/s]': heave_rate}, index=rv_meteor.index[1:])
rv_meteor = rv_meteor.join(heave_rate)
print(f"Done with heave rate calculation in {time.time() - t1}")

# calculate average heave rate for chirp time length 500 ms
rv_meteor_500ms = rv_meteor.resample('0.5S').mean()

########################################################################################################################
# Plotting section
########################################################################################################################
# general plotting options
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16, 'figure.figsize': (16, 9)})

########################################################################################################################
# PDF
var = 'Heave Rate [m/s]'
# plot PDF, 40 bins
plt.hist(rv_meteor_500ms[var],
         bins=40, density=True, histtype='bar', log=True, label=["0.5s"], color='b')
plt.legend(title="Sampling Frequency")
plt.xlabel(f"{var}")
plt.ylabel("Probability Density")
# plt.show()
if end_dt.date() > begin_dt.date():
    plt.title(f"Probability Density Function of {var[:-6]} - DSHIP\n"
              f" EUREC4A RV-Meteor {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}")
    plt.tight_layout()
    filename = f"{output_path}RV-Meteor_seapath_{var[:-6]}_PDF_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}_log.png"
    plt.savefig(filename, dpi=250)
    print(f"{filename} saved to {output_path}")
else:
    plt.title(f"Probability Density Function of {var[:-6]} - DSHIP\n"
              f" EUREC4A RV-Meteor {begin_dt:%Y-%m-%d}")
    plt.tight_layout()
    filename = f"{output_path}RV-Meteor_seapath_{var[:-6]}_PDF_{begin_dt:%Y%m%d}_log.png"
    plt.savefig(filename, dpi=250)
    print(f"{filename} saved to {output_path}")
plt.close()

########################################################################################################################
# Timeseries
# set date tick locator and formater
hfmt = mdates.DateFormatter('%d/%m/%y %H')

fig, ax = plt.subplots()
ax.plot(rv_meteor['Heave Rate [m/s]'], 'g', label='Heave Rate')
ax.set_ylabel("Heave Rate [m/s]")
ax.set_xlabel("Datetime [UTC]")
ax.xaxis.set_major_formatter(hfmt)
ax.legend()
fig.autofmt_xdate()
if end_dt.date() > begin_dt.date():
    ax.set_title(f"Time Series of {var[:-6]} - DSHIP\n"
              f" EUREC4A RV-Meteor {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}")
    fig.tight_layout()
    filename = f"{output_path}RV-Meteor_seapath_{var[:-6]}_TS_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}.png"
    fig.savefig(filename, dpi=250)
    print(f"{filename} saved to {output_path}")
else:
    ax.set_title(f"Time Series of {var[:-6]} - DSHIP\n"
              f" EUREC4A RV-Meteor {begin_dt:%Y-%m-%d}")
    fig.tight_layout()
    filename = f"{output_path}RV-Meteor_seapath_{var[:-6]}_TS_{begin_dt:%Y%m%d}.png"
    fig.savefig(filename, dpi=250)
    print(f"{filename} saved to {output_path}")
plt.close()