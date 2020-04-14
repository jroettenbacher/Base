#!/bin/python3
''' Hydrometeor Fraction
make hydrometeor fraction for all of eurec4a
need to interpolate the range gates between first part and second part of the campaign
'''
import numpy as np
import pandas as pd
import sys
import time
from scipy import interpolate
# just needed to find pyLARDA from this location
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda/')
sys.path.append('.')
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FormatStrFormatter)
import pyLARDA
import pyLARDA.helpers as h
import datetime
import logging

log = logging.getLogger('pyLARDA')
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler())

# Load LARDA
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)

########################################################################################################################
# Set Date and Paths
########################################################################################################################
begin_dt = [datetime.datetime(2020, 1, 17, 0, 0, 5), datetime.datetime(2020, 1, 29, 18, 1, 0),
            datetime.datetime(2020, 1, 30, 15, 8, 5), datetime.datetime(2020, 1, 31, 22, 27, 0)]
end_dt = [datetime.datetime(2020, 1, 29, 17, 59, 55), datetime.datetime(2020, 1, 30, 15, 7, 55),
          datetime.datetime(2020, 1, 31, 22, 26, 50), datetime.datetime(2020, 2, 19, 23, 59, 55)]
# calculate hours for weighted average
hours = dict()
i = 0
for begin, end in zip(begin_dt, end_dt):
    i += 1
    hours[i] = (end - begin).total_seconds() / 60 / 60

# define path where to write csv file (no / at end of path please)
# output_path = "/home/remsens/code/larda3/scripts/plots/radar_hydro_frac"
output_path = "/projekt1/remsens/work/jroettenbacher/Base/tmp"
# define output path for plots (no / at end of path please)
# plot_path = "/home/remsens/code/larda3/scripts/plots/radar_hydro_frac"
plot_path = "/projekt1/remsens/work/jroettenbacher/plots"

########################################################################################################################
# read in files with larda
########################################################################################################################
i = 0
hydro_out = dict()
for begin, end in zip(begin_dt, end_dt):
    i += 1
    print(f"Starting with Ze{i}\n")
    print("Read in data...\n")
    Ze = larda.read("LIMRAD94_cn_input", "Ze", [begin, end], [0, 'max'])
    t1 = time.time()
    # mask values = -999
    Ze["var"] = np.ma.masked_where(Ze["var"] == -999, Ze["var"])
    # overwrite mask in larda container
    Ze["mask"] = Ze["var"].mask

    ####################################################################################################################
    # extract variables from container
    ####################################################################################################################
    range_bins = Ze['rg']  # unit m

    ####################################################################################################################
    # calculate hydrometeor fraction
    ####################################################################################################################
    # allocate array for all CFs but the first row (ghost echo bug fix)
    print("Calculating hydrometeor fraction\n")
    Ze_CF = np.empty((Ze['var'].shape[1] - 1))
    for j in range(0, Ze_CF.shape[0]):
        # loop through all height bins but the first one
        # mean hydrometeor fraction profile for whole time range
        # sum up all values which have a reflectivity value and divide by the number of time steps
        Ze_CF[j] = np.sum(~Ze['mask'][:, j + 1]) / Ze['var'].shape[0]

    ####################################################################################################################
    # create data frame with named columns and range bins as index column, round hydrometeor fractions to 4 decimals
    ####################################################################################################################
    # create column labels
    columns = ['hydro_frac']
    # create data frame for plotting
    hydro_frac = pd.DataFrame(data=np.round(Ze_CF, decimals=4),
                              columns=columns,
                              index=np.floor(range_bins[1:len(Ze_CF)+1]))
    hydro_frac.index.name = "Height_m"  # set index title of data frame
    # save data frame to dictionary
    hydro_out[f'Ze{i}'] = hydro_frac.reset_index()

    # write csv file
    print(f"Writing csv file...\n")
    hydro_frac.to_csv(f"{output_path}/hydro_frac_Ze{i}.csv", sep=",", na_rep="NA")
    print(f"Done with Ze{i} in {time.time() - t1:.3f} seconds\n")
########################################################################################################################
# interpolation between different range resolution
########################################################################################################################
print("Interpolating data\n")
new_hydro = dict()
for i in range(len(begin_dt)-1):
    j = i + 1
    # create linear function to model hydro fractions
    f = interpolate.interp1d(hydro_out[f"Ze{j}"]["Height_m"], hydro_out[f"Ze{j}"]["hydro_frac"], kind='linear')
    try:
        new_hydro[j] = f(hydro_out["Ze4"]["Height_m"])  # interpolate data to height bins of last time range
    except ValueError:
        new_height_bins = hydro_out["Ze4"]["Height_m"][hydro_out["Ze4"]["Height_m"] <
                                                       np.max(hydro_out[f"Ze{j}"]["Height_m"])]
        new_hydro[j] = f(new_height_bins)  # interpolate data to height bins of last time range
        # add nan values to new_hydro if it is shorter than Ze4
        needed_values = len(hydro_out["Ze4"]["Height_m"]) - len(new_hydro[j])
        fill_array = np.empty(needed_values)
        fill_array.fill(np.nan)
        new_hydro[j] = np.append(new_hydro[j], fill_array)


# combine hydro fractions by a weighted average, weighted by hours
new_hydro_frac = (new_hydro[1] * hours[1] + new_hydro[2] * hours[2] + new_hydro[3] * hours[3]
                  + hydro_out["Ze4"]["hydro_frac"] * hours[4]) / 4 / sum(hours.values())
hydro_frac = pd.DataFrame({'Height_m': hydro_out["Ze4"]["Height_m"], "hydro_frac": new_hydro_frac})
########################################################################################################################
# plotting section
########################################################################################################################
print("Start plotting...")
# some layout stuff
plt.style.use("default")
plt.rcParams.update({'font.size': 16, 'figure.figsize': (10, 10)})

fig, ax = plt.subplots()
ax.plot(hydro_frac['hydro_frac'], hydro_frac['Height_m'], label="Hydrometeor Fraction", linewidth=3)
ax.legend(title='', fontsize=14, bbox_to_anchor=(1., 1.))
ax.set_ylabel("Height [m]")
ax.set_xlabel("Hydrometeor Fraction")
ax.set_title(
    f"Hydrometeor Fraction in the whole Troposphere Eurec4a "
    f"\n RV-Meteor - {begin_dt[0]:%Y-%m-%d} - {end_dt[-1]:%Y-%m-%d} "
    f"\nCloudradar Uni Leipzig")
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_formatter(FormatStrFormatter('%d'))
ax.tick_params(which='minor', length=4, labelsize=12)
plt.tight_layout()
ax.grid(True, which='minor', color="grey", linestyle='-', linewidth=1)
ax.xaxis.grid(True, which='major', color="k", linestyle='-', linewidth=2, alpha=0.5)
ax.yaxis.grid(True, which='major', color="k", linestyle='-', linewidth=2, alpha=0.5)
# plt.show()
plt.savefig(f"{plot_path}/RV-Meteor_cloudradar_hydro-fraction_{begin_dt[0]:%Y%m%d}-{end_dt[-1]:%Y%m%d}.png", dpi=250)
print(f"Saved figure to {plot_path}")

print("Done with script.")
