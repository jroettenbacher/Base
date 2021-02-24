#!/bin/python3
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FormatStrFormatter)
import datetime
import functions_jr as jr

# goal: for daily eurec4a files, take Ze and divide it into chunks of 3hrs.
# For each 3hr time-height chunk and each range between first range gate and 1200m (eurec4a: range gate 41),
# determine the cloud fraction (number of profiles with Ze divided by number of profiles in 3hrs
# then find the altitude (below 1200m) with the max. Cloud fraction (CF) for each 3hr time chunk
# output these values as a text file with the times 0-3 UTC, 4-6 UTC etc as header

# to be added:
# standard deviation: day (24h), 3h ( 1 hourly data)
# mean cloud fraction profile plots (24h) with standard deviation
# mean Ze just above cloud base per 3h -> problem rain! throw out rainy data or use ceilometer
# plot: all time chunks in one plot x=hydrometeor fraction, y=height, check JR

########################################################################################################################
# Loading Data
########################################################################################################################
# command line input from user

kwargs = jr.read_command_line_args()
date = kwargs['date'] if 'date' in kwargs else '20200209'
chunk_size = kwargs['chunk_size'] if 'chunk_size' in kwargs else 1  # chunk size in hours
print(f"Running quick_cloud_stat.py for date={date} and chunk_size={chunk_size:.0f}")

dt_date = datetime.datetime.strptime(date, "%Y%m%d")  # transform to datetime for plot title
# load data from Catalpa
# path = "/home/remsens/data/LIMRAD94/cloudnet_input"
# load data from LIM server
path = "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/LIMRAD94/cloudnet_input"
# load local data
# path = "./data/cloudnet_input/"

# define path where to write csv file (no / at end of path please)
output_path = "/home/remsens/code/larda3/scripts/plots/radar_hydro_frac"
# local output path
# output_path = "./tmp"

# define output path for plots (no / at end of path please)
plot_path = "/home/remsens/code/larda3/scripts/plots/radar_hydro_frac"
# local path
# plot_path = "./tmp"

# define upper ranges for lower, mid and high troposphere in meters
low, mid, high = 1200, 6000, 12000

# read in nc file (adjust file name)
limrad94_ds = nc.Dataset(f"{path}/{date}_000000-240000_LIMRAD94.nc")

########################################################################################################################
# extract variables from nc files (adjust variable names)
########################################################################################################################
nc_time = nc.num2date(limrad94_ds.variables["time"][:],
                      units="seconds since 2001-01-01 00:00:00 UTC",
                      calendar="standard")
# get length of file timewise, first as datetime.timedelta, extract seconds, convert to decimal hours
# and round to the next full hour
hours = np.ceil((max(nc_time) - min(nc_time)).seconds / 3600)
Ze = limrad94_ds.variables["Ze"][:]  # unit [mm6/m3], dimensions : time, range
range_bins = limrad94_ds.variables["range"][:]  # unit m
# get indices for corresponding low mid and high upper ranges
ind = [0]  # start with zero to define lower boundary
for i in [low, mid, high]:
    ind.append(np.where(np.abs(range_bins-i) == np.min(np.abs(range_bins-i)))[0][0])

########################################################################################################################
# calculate hydrometeor fraction for time chunks for the low, mid and high atmosphere
########################################################################################################################
num_chunks = int(hours / chunk_size)  # get number of chunks by dividing number of observed hours by chunk size
time_chunk = int(np.floor(len(nc_time) / num_chunks))  # get rounded down number of profiles in each time chunk
Ze_CF = np.empty((len(range_bins), num_chunks))
Ze_CF_max_low = np.empty((2, num_chunks))
Ze_CF_max_low_mid_high = np.empty((6, num_chunks))
Ze_CF_max_all = np.empty((2, num_chunks))

for i in range(0, Ze_CF.shape[1]):
    # loop through all time chunks
    # select all rows (height bins) but only the columns (time) of the current time chunk
    # this has an overlap of one time step
    Ze_low_3h = Ze[slice(time_chunk * i, time_chunk * (i + 1)), :]
    for j in range(0, Ze_CF.shape[0]):
        # loop though all rows (height bins)
        # mean hydrometeor fraction profile per 3h chunk
        # sum up all values which have a reflectivity value and divide by the number of time steps
        Ze_CF[j, i] = np.sum(~Ze_low_3h[j, :].mask) / Ze_low_3h.shape[1]
    # TODO standard deviation of hydrometeor fraction per 3h chunk
    # find index of highest hydrometeor fraction
    index = np.where(Ze_CF[:, i] == np.max(Ze_CF[:, i]))[0][0]
    Ze_CF_max_all[0, i] = range_bins[index]  # use range bin instead of index
    Ze_CF_max_all[1, i] = np.max(Ze_CF[:, i])  # find value of highest hydrometeor fraction

    # for low, mid and high troposphere
    for k in range(len(ind)-1):
        # find index of highest hydrometeor fraction in low, mid and high troposphere
        index = np.where(Ze_CF[ind[k]:ind[k+1], i] == np.max(Ze_CF[ind[k]:ind[k+1], i]))[0][0] + ind[k]
        Ze_CF_max_low_mid_high[2*k, i] = range_bins[index]  # use range bin instead of index
        Ze_CF_max_low_mid_high[1+(2*k), i] = np.max(Ze_CF[ind[k]:ind[k+1], i])  # find value of highest hydrometeor fraction

# cut off all range bins above low range border
Ze_CF_out = Ze_CF[0:ind[1], :]

########################################################################################################################
# create data frame with named columns and range bins as index column, round hydrometeor fractions to 2 decimals
########################################################################################################################
# create column labels depending on chunk size and number of chunks
# need as many labels as number of chunks
# initiate list with column names, first name always starts with 0UTC
# special case for one hour chunks
if chunk_size == 1:
    columns = []
    for i in range(num_chunks):
        columns.append(f"{i}UTC")
else:
    columns = [f"0-{chunk_size:.0f}UTC"]
    for i in range(2, num_chunks + 1):
        # append labels
        columns.append(f"{chunk_size * (i - 1) + 1:.0f}-{chunk_size * i:.0f}UTC")

hydro_frac = pd.DataFrame(data=np.round(Ze_CF_out, decimals=2),
                          columns=columns,
                          index=np.floor(range_bins[0:len(Ze_CF_out)]))
hydro_frac.index.name = "Height_m"  # set index title of data frame
# write data frame (table) to csv file, separator= tabstop, display 2 decimal places
# hydro_frac.to_csv(f"{output_path}/{date}_hydro_fracs_RV-Meteor_{chunk_size:.0f}h.csv",
#                   sep='\t', float_format="%.2f")

########################################################################################################################
# maximal hydrometeor fraction per time chunk in lower troposphere
########################################################################################################################
# max_hydro_frac_low = pd.DataFrame(data=np.round(Ze_CF_max_low, decimals=2),
#                                   columns=columns,
#                                   index=["Height_m", "max_hydro_frac"])
# max_hydro_frac_low.index.name = "Time"  # set index title of data frame to label header
# # write data frame (table) to csv file
# max_hydro_frac_low.to_csv(f"{output_path}/{date}_max_hydro_frac_RV-Meteor_low_{chunk_size:.0f}h.csv",
#                           sep='\t', float_format="%.2f")
#
########################################################################################################################
# maximal hydrometeor fraction per time chunk in low, mid level and upper level troposphere
########################################################################################################################
max_hydro_frac_low_mid_high = pd.DataFrame(data=np.round(Ze_CF_max_low_mid_high, decimals=2),
                                           columns=columns,
                                           index=["Height_m", "max_hydro_frac_low", "Height_m", "max_hydro_frac_mid",
                                                  "Height_m", "max_hydro_frac_high"])
max_hydro_frac_low_mid_high.index.name = "Time"  # set index title of data frame to label header
# write data frame (table) to csv file
max_hydro_frac_low_mid_high.to_csv(f"{output_path}/{date}_max_hydro_frac_RV-Meteor_low_mid_high_{chunk_size:.0f}hr.txt",
                                   sep='\t', float_format="%.2f")

########################################################################################################################
# maximal hydrometeor fraction per time chunk whole troposphere
########################################################################################################################
# max_hydro_frac_all = pd.DataFrame(data=np.round(Ze_CF_max_all, decimals=2),
#                                   columns=columns,
#                                   index=["Height_m", "max_hydro_frac"])
# max_hydro_frac_all.index.name = "Time"  # set index title of data frame to label header
# # write data frame (table) to csv file
# max_hydro_frac_all.to_csv(f"{output_path}/{date}_max_hydro_frac_RV-Meteor_all_{chunk_size:.0f}h.csv",
#                           sep='\t', float_format="%.2f")
#
print(f"csv files saved to {output_path}")

########################################################################################################################
# plotting section
########################################################################################################################
print("Start plotting...")
# some layout stuff
plt.style.use("default")
plt.rcParams.update({'font.size': 16, 'figure.figsize': (16, 9)})

# easy but wrong axes
# hydromet_frac.plot()
# plt.show()
# more text but right axes
hydro_frac_plt = hydro_frac.reset_index()

cm = plt.get_cmap('tab20')  # get colormap
fig, ax = plt.subplots()
# define colors to choose from for each line
ax.set_prop_cycle(color=[cm(1 * i / num_chunks) for i in range(num_chunks)])
for i in range(num_chunks):
    ax.plot(hydro_frac_plt.iloc[:, i + 1], hydro_frac_plt['Height_m'], label=columns[i], linewidth=3)
ax.legend(title='Time', fontsize=14, bbox_to_anchor=(1., 1.))
ax.set_ylabel("Height [m]")
ax.set_xlabel("Hydrometeor Fraction")
ax.set_title(
    f"Hydrometeor Fraction in the Lower Troposphere Eurec4a \n RV-Meteor - {dt_date:%Y-%m-%d} \nCloudradar Uni Leipzig")
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_formatter(FormatStrFormatter('%d'))
ax.tick_params(which='minor', length=4, labelsize=12)
plt.tight_layout()
ax.grid(True, which='minor', color="grey", linestyle='-', linewidth=1)
ax.xaxis.grid(True, which='major', color="k", linestyle='-', linewidth=2, alpha=0.5)
ax.yaxis.grid(True, which='major', color="k", linestyle='-', linewidth=2, alpha=0.5)
# plt.show()
plt.savefig(f"{plot_path}/{date}_hydro_frac_RV-Meteor_{chunk_size:.0f}hr.png", dpi=75)
print(f"Saved figure to {plot_path}")
