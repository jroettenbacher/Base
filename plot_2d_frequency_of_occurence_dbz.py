#!/usr/bin/env python
"""script to plot a height resolved frequency of occurence plot of dBZ values
input: Ze from larda
output: 2d frequency of occurence plot
author: Johannes Roettenbacher
"""

import sys
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import pyLARDA
import pyLARDA.helpers as h
import datetime as dt
import time
import numpy as np
import functions_jr as jr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import IndexFormatter

start = time.time()
plot_path = "../plots"

# load LARDA
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)

# define larda stuff
system = "LIMRAD94_cn_input"
begin_dt = dt.datetime(2020, 2, 1, 0, 0, 5)
end_dt = dt.datetime(2020, 2, 19, 23, 59, 55)
plot_range = [0, 'max']

# read in dBZ values from cloudnet input
print("Reading in data from larda")
radar_ze = larda.read(system, "Ze", [begin_dt, end_dt], plot_range)

print("Computing histograms")
t1 = time.time()
# initialize a list to hold each histogram
hist = []
# calculate histogramms for each height bin, remove nans beforehand
# define histogram options
hist_bins = np.arange(-59, 31, 1)  # bin edges including the rightmost bin
hist_range = (-61, 30)  # upper and lower bin boundary
density = False  # compute the probability density function
for i in range(radar_ze['var'].shape[1]):
    h_bin = h.lin2z(radar_ze['var'][:, i])  # select height bin and convert to dBZ
    mask = radar_ze['mask'][:, i]  # get mask for height bin
    h_bin[mask] = np.nan  # set -999 to nan
    tmp_hist, _ = np.histogram(h_bin[~np.isnan(h_bin)], bins=hist_bins, range=hist_range, density=density)
    hist.append(tmp_hist)

# create an array to plot with pcolormesh by stacking the computed histograms vertically
plot_array = np.array(hist, dtype=np.float)
# set zeros to nan
plot_array[plot_array == 0] = np.nan
# normalize by count of all pixels
plot_array = plot_array / np.nansum(plot_array)
print(f"Computed histograms and created array for plotting in {time.time() - t1:.2f} seconds")

# get mean sensitivity limit
mean_sl = jr.calc_sensitivity_curve(['P07'])
mean_slh_p07 = mean_sl['mean_slh']['P07']
mean_slv_p07 = mean_sl['mean_slv']['P07']
# scale sensitivity limit to pcolormesh axis = 0-89, lowest value in histogram bins
mean_slh_p07 = mean_slh_p07 + np.abs(hist_bins[0])
mean_slv_p07 = mean_slv_p07 + np.abs(hist_bins[0])

########################################################################################################################
# plotting section
########################################################################################################################
figname = f"{plot_path}/tmp.png"

# create an array for the x and y tick labels
height = radar_ze['rg']
ylabels = np.floor(height) // 100 * 100
xlabels = hist_bins - 1

# create title
program_names = {'P09': "tradewindCU (P09)", 'P06': "Cu_small_Tint (P06)", 'P07': "Cu_small_Tint2 (P07)"}
title = f"Frequency of Occurrence of Reflectivity " \
        f"\nEUREC4A {begin_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}" \
        f"\n Chirp program: {program_names['P07']}"

fig, ax = plt.subplots()
im = ax.pcolormesh(plot_array, cmap='jet', norm=LogNorm())
ax.plot(mean_slh_p07, np.arange(len(height)), "k-", label="Sensitivity Limit")
fig.colorbar(im, ax=ax)
fig.legend()
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.xaxis.set_major_formatter(IndexFormatter(xlabels))
ax.yaxis.set_major_formatter(IndexFormatter(ylabels))
ax.tick_params(which='minor', length=4)
ax.tick_params(which='major', length=6)
ax.set_xlabel("Reflectivity [dBZ]")
ax.set_ylabel("Height [m]")
ax.set_title(title)
fig.savefig(figname)
plt.close()
