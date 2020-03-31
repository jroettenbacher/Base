#!/bin/python3
''' Hydrometeor Fraction
make hydrometeor fraction for all of eurec4a
need to interpolate the range gates between first part and second part of the campaign
'''
import netCDF4 as nc
import numpy as np
import pandas as pd
import sys
# just needed to find pyLARDA from this location
sys.path.append('/home/remsens/code/larda3/larda/')
sys.path.append('.')
import matplotlib
matplotlib.use('Agg')
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
begin_dt = datetime.datetime(2020, 1, 18, 0, 0, 5)
end_dt = datetime.datetime(2020, 1, 31, 23, 59, 55)
begin_dt2 = datetime.datetime(2020, 2, 1, 0, 0, 5)
end_dt2 = datetime.datetime(2020, 2, 19, 23, 59, 55)
# if chunk size is smaller 12: x hourly hydro fractions are calculated and plotted -> should be used for one day only
# if chunk size is greater 11 the mean, median and std of the x hr chunks are calculated and plotted
# if chunk size is 'max' then the hydrometer fraction over the whole period is calculated and plotted (like BAMS paper)
chunk_size = 'max'  # chunk size in hours

# define path where to write csv file (no / at end of path please)
# output_path = "/home/remsens/code/larda3/scripts/plots/radar_hydro_frac"
output_path = "/project1/remsens/work/jroettenbacher/plots"
# define output path for plots (no / at end of path please)
# plot_path = "/home/remsens/code/larda3/scripts/plots/radar_hydro_frac"
plot_path = "/project1/remsens/work/jroettenbacher/plots"

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
    # # define upper ranges for lower, mid and high troposphere in meters
    # low, mid, high = 1200, 6000, np.ceil(range_bins.max())
    # # get indices for corresponding low mid and high upper ranges
    # ind = [0]  # start with zero to define lower boundary
    # # quick bug fix
    # # start with 1 to ignore ghost echo in first range gate which is caused by the new chirp table Cu_small_Tint2
    # ind = [1]
    # for i in [low, mid, high]:
    #     ind.append(np.where(np.abs(range_bins-i) == np.min(np.abs(range_bins-i)))[0][0])

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

########################################################################################################################
# interpolation between different range resolution
########################################################################################################################


########################################################################################################################
# plotting section
########################################################################################################################
print("Start plotting...")
# some layout stuff
plt.style.use("default")
plt.rcParams.update({'font.size': 16, 'figure.figsize': (10, 10)})

if chunk_size == 'max':
    hydro_frac_plt = hydro_frac.reset_index()

    fig, ax = plt.subplots()
    ax.plot(hydro_frac_plt['hydro_frac'], hydro_frac_plt['Height_m'], label="Hydrometeor Fraction", linewidth=3)
    ax.legend(title='', fontsize=14, bbox_to_anchor=(1., 1.))
    ax.set_ylabel("Height [m]")
    ax.set_xlabel("Hydrometeor Fraction")
    ax.set_title(
        f"Hydrometeor Fraction in the whole Troposphere Eurec4a "
        f"\n RV-Meteor - {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d} "
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
    plt.savefig(f"{plot_path}/RV-Meteor_cloudradar_hydro-fraction_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}.png", dpi=250)
    print(f"Saved figure to {plot_path}")

elif chunk_size < 12:
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
        f"Hydrometeor Fraction in the whole Troposphere Eurec4a "
        f"\n RV-Meteor - {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d} "
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
    plt.savefig(f"{plot_path}/RV-Meteor_cloudradar_{chunk_size}hourly_hydro-fraction_{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}.png", dpi=250)
    print(f"Saved figure to {plot_path}")

elif chunk_size >= 12:
    hydro_frac_plt = hydro_frac_stat.reset_index().iloc[::5, :]

    fig, ax = plt.subplots()
    # define colors to choose from for each line
    # ax.plot(hydro_frac_plt['Mean'], hydro_frac_plt['Height_m'], label="Mean Hydrometeor Fraction", linewidth=3)
    ax.errorbar(hydro_frac_plt['Mean'], hydro_frac_plt['Height_m'], xerr=hydro_frac_plt['Std'], fmt='-b',
                ecolor='k', label="Mean Hydrometeor Fraction")  #, linewidth=2)
    # ax.legend(title='', fontsize=14, bbox_to_anchor=(1., 1.))
    ax.set_ylabel("Height [m]")
    ax.set_xlabel("Hydrometeor Fraction")
    ax.set_title(f"{chunk_size}h Mean Hydrometeor Fraction in the whole Troposphere Eurec4a "
                 f"\n RV-Meteor - {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d} "
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
    plt.savefig(f"{plot_path}/RV-Meteor_cloudradar_{chunk_size}hourly_mean_hydro-fraction_"
                f"{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}.png", dpi=250)
    plt.close()
    print(f"Saved figure of mean hydro-frac to {plot_path}")

    fig, ax = plt.subplots()
    # define colors to choose from for each line
    # ax.plot(hydro_frac_plt['Mean'], hydro_frac_plt['Height_m'], label="Mean Hydrometeor Fraction", linewidth=3)
    ax.errorbar(hydro_frac_plt['Median'], hydro_frac_plt['Height_m'], xerr=hydro_frac_plt['Std'], fmt='-b',
                ecolor='k', label="Median Hydrometeor Fraction")  # , linewidth=2)
    # ax.legend(title='', fontsize=14, bbox_to_anchor=(1., 1.))
    ax.set_ylabel("Height [m]")
    ax.set_xlabel("Hydrometeor Fraction")
    ax.set_title(f"Median Hydrometeor Fraction in the whole Troposphere Eurec4a "
                 f"\n RV-Meteor - {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d} "
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
    plt.savefig(f"{plot_path}/RV-Meteor_cloudradar_{chunk_size}hourly_median_hydro-fraction_"
                f"{begin_dt:%Y%m%d}-{end_dt:%Y%m%d}.png", dpi=250)
    plt.close()
    print(f"Saved figure of median hydro-frac to {plot_path}")
