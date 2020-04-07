#!/bin/python3
''' Hydrometeor Fraction
make hydrometeor fraction for all of eurec4a
need to interpolate the range gates between first part and second part of the campaign
'''
import numpy as np
import pandas as pd
import sys
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
# Set Date, Chunk Size and Paths
########################################################################################################################
begin_dt = datetime.datetime(2020, 1, 17, 0, 0, 5)
end_dt = datetime.datetime(2020, 1, 31, 23, 59, 55)
begin_dt2 = datetime.datetime(2020, 2, 1, 0, 0, 5)
end_dt2 = datetime.datetime(2020, 2, 19, 23, 59, 55)

# if chunk size is 'max' then the hydrometer fraction over the whole period is calculated and plotted (like BAMS paper)
chunk_size = 'max'  # chunk size in hours

# define path where to write csv file (no / at end of path please)
# output_path = "/home/remsens/code/larda3/scripts/plots/radar_hydro_frac"
output_path = "/project1/remsens/work/jroettenbacher/Base/tmp"
# define output path for plots (no / at end of path please)
# plot_path = "/home/remsens/code/larda3/scripts/plots/radar_hydro_frac"
plot_path = "/project1/remsens/work/jroettenbacher/Base/plots"

########################################################################################################################
# read in files with larda
########################################################################################################################
Ze1 = larda.read("LIMRAD94_cn_input", "Ze", [begin_dt, end_dt], [0, 'max'])
Ze2 = larda.read("LIMRAD94_cn_input", "Ze", [begin_dt2, end_dt2], [0, 'max'])
i = 0
hydro_out = dict
for Ze in [Ze1, Ze2]:
    i += 1
    # mask values = -999
    Ze["var"] = np.ma.masked_where(Ze["var"] == -999, Ze["var"])
    # overwrite mask in larda container
    Ze["mask"] = Ze["var"].mask

    ####################################################################################################################
    # extract variables from container
    ####################################################################################################################
    range_bins = Ze['rg']  # unit m
    time_dt = np.asarray([h.ts_to_dt(ts) for ts in Ze['ts']])
    # get length of file timewise, first as datetime.timedelta, extract seconds, convert to decimal hours
    # and round to the next full hour
    hours = np.ceil((time_dt.max() - time_dt.min()).total_seconds() / 3600)

    ####################################################################################################################
    # calculate hydrometeor fraction
    ####################################################################################################################
    # allocate array for all CFs but the first row (ghost echo bug fix)
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
    hydro_out[f'Ze{i}'] = hydro_frac

    # write csv file
    hydro_frac.to_csv(f"{output_path}/hydro_frac_Ze{i}.csv", sep=",", na_rep="NA")
########################################################################################################################
# interpolation between different range resolution
########################################################################################################################


########################################################################################################################
# plotting section
########################################################################################################################
# print("Start plotting...")
# # some layout stuff
# plt.style.use("default")
# plt.rcParams.update({'font.size': 16, 'figure.figsize': (10, 10)})
#
# if chunk_size == 'max':
#     hydro_frac_plt = hydro_frac.reset_index()
#
#     fig, ax = plt.subplots()
#     ax.plot(hydro_frac_plt['hydro_frac'], hydro_frac_plt['Height_m'], label="Hydrometeor Fraction", linewidth=3)
#     ax.legend(title='', fontsize=14, bbox_to_anchor=(1., 1.))
#     ax.set_ylabel("Height [m]")
#     ax.set_xlabel("Hydrometeor Fraction")
#     ax.set_title(
#         f"Hydrometeor Fraction in the whole Troposphere Eurec4a "
#         f"\n RV-Meteor - {begin_dt:%Y-%m-%d} - {end_dt2:%Y-%m-%d} "
#         f"\nCloudradar Uni Leipzig")
#     ax.yaxis.set_minor_locator(AutoMinorLocator(5))
#     ax.xaxis.set_minor_locator(AutoMinorLocator(2))
#     ax.yaxis.set_minor_formatter(FormatStrFormatter('%d'))
#     ax.tick_params(which='minor', length=4, labelsize=12)
#     plt.tight_layout()
#     ax.grid(True, which='minor', color="grey", linestyle='-', linewidth=1)
#     ax.xaxis.grid(True, which='major', color="k", linestyle='-', linewidth=2, alpha=0.5)
#     ax.yaxis.grid(True, which='major', color="k", linestyle='-', linewidth=2, alpha=0.5)
#     # plt.show()
#     plt.savefig(f"{plot_path}/RV-Meteor_cloudradar_hydro-fraction_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}.png", dpi=250)
#     print(f"Saved figure to {plot_path}")
