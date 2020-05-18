#!/usr/bin/python

"""plot sensitivity curve for each chirp of LIMRAD94
sensitivity limit is calculated for horizontal and vertical polarization
for more information see LIMRAD94 manual chapter 2.6
"""


import sys
# just needed to find pyLARDA from this location
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pyLARDA
import pyLARDA.helpers as h
import datetime as dt
import numpy as np
import pandas as pd
import logging

log = logging.getLogger('pyLARDA')
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler())

# define plot path
plot_path = "/projekt1/remsens/work/jroettenbacher/plots/sensitivity"
# Load LARDA
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
system = "LIMRAD94"
begin_dt = dt.datetime(2020, 1, 20, 0, 0, 5)
end_dt = dt.datetime(2020, 1, 20, 23, 59, 55)
plot_range = [0, 'max']

# read in sensitivity variables over whole range (all chirps)
slv = larda.read(system, "SLv", [begin_dt, end_dt], plot_range)
slh = larda.read(system, "SLh", [begin_dt, end_dt], plot_range)

# decide which chirptable to add to plot title
chirptables = ("tradewindCU (P09)", "Cu_small_Tint (P06)", "Cu_small_Tint2 (P07)")
if begin_dt.date() in pd.date_range(dt.datetime(2020, 1, 16), dt.datetime(2020, 1, 29)):
    chirptable = chirptables[0]
elif begin_dt.date() in pd.date_range(dt.datetime(2020, 1, 30, 15, 8), dt.datetime(2020, 1, 31, 22, 27)):
    chirptable = chirptables[1]
else:
    chirptable = chirptables[2]

# get range bins at chirp borders
ranges = {}
no_chirps = 3
range_bins = np.zeros(no_chirps + 1, dtype=np.int)  # needs to be length 4 to include all +1 chirp borders
for i in range(no_chirps):
    ranges[f'C{i + 1}Range'] = larda.read(system, f'C{i + 1}Range', [begin_dt, end_dt])
    try:
        range_bins[i + 1] = range_bins[i] + ranges[f'C{i + 1}Range']['var'][0].shape
    except ValueError:
        # in case only one file is read in data["C1Range"]["var"] has only one dimension
        range_bins[i + 1] = range_bins[i] + ranges[f'C{i + 1}Range']['var'].shape

# time height plot of sensitivity for all chirps
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_sensitivity'
fig, _ = pyLARDA.Transformations.plot_timeheight(slv, rg_converter=False, title=True, z_converter='lin2z')
fig.savefig(f'{name}_allChirps_{chirptable}.png', dpi=250)
print(f'figure saved :: {name}_allChirps_{chirptable}.png')
plt.close()

# plotting a sensitivity curve = mean sensitivity during measurement interval
slv_c1 = np.mean(slv["var"][:, range_bins[0]:range_bins[1]], axis=0)
slv_c2 = np.mean(slv["var"][:, range_bins[1]:range_bins[2]], axis=0)
slv_c3 = np.mean(slv["var"][:, range_bins[2]:range_bins[3]], axis=0)
slh_c1 = np.mean(slh["var"][:, range_bins[0]:range_bins[1]], axis=0)
slh_c2 = np.mean(slh["var"][:, range_bins[1]:range_bins[2]], axis=0)
slh_c3 = np.mean(slh["var"][:, range_bins[2]:range_bins[3]], axis=0)
heights = {}
for i in range(len(ranges)):
    heights[f"C{i+1}Range"] = ranges[f"C{i+1}Range"]["var"][1, :]
plt.style.use('default')
plt.rcParams.update({'figure.figsize': (16, 9)})

# plot with all chirps and both polarizations
fig, axs = plt.subplots(ncols=no_chirps, constrained_layout=True)
for ax, slv, slh, i in zip(axs, [slv_c1, slv_c2, slv_c3], [slh_c1, slh_c2, slh_c3], [1, 2, 3]):
    ax.plot(h.lin2z(slv), heights[f"C{i}Range"], label='vertical')
    ax.plot(h.lin2z(slh), heights[f"C{i}Range"], label='horizontal')
    ax.set_title(f"Chirp {i}")
    ax.set_ylabel("Height [m]")
    ax.set_xlabel("Sensitivity Limit [dBZ]")
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.grid(True, which='both', axis='both', color="grey", linestyle='-', linewidth=1)
    ax.legend(title="Polarization")
fig.suptitle(f"Mean Sensitivity limit for LIMRAD94 \n"
             f"Eurec4a - {begin_dt:%Y-%m-%d %H:%M} to {end_dt:%Y-%m-%d %H:%M} UTC\n"
             f"Chirp Table: {chirptable}", fontsize=16)
fig.savefig(f'{name}_curves.png', dpi=250)
# fig.subplots_adjust(top=0.7, wspace=0.6)
# fig.tight_layout()
print(f'figure saved :: {name}_curves_.png')
plt.close()

# one plot for each curve
for var, i in zip([slv_c1, slv_c2, slv_c3], (1, 2, 3)):
    plt.plot(h.lin2z(var), heights[f"C{i}Range"])
    plt.title(f"Mean Sensitivity limit for LIMRAD94 \n- Chirp {i} vertical polarization -\n"
              f"Eurec4a - {begin_dt:%Y-%m-%d %H:%M} to {end_dt:%Y-%m-%d %H:%M} UTC\n"
              f"Chirptable: {chirptable}")
    plt.ylabel("Height [m]")
    plt.xlabel("Sensitivity Limit [dBZ]")
    plt.minorticks_on()
    plt.axes().xaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.axes().yaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.grid(True, which='both', axis='both', color="grey", linestyle='-', linewidth=1)
    plt.savefig(f'{name}_Chirp{i}_curve_vertical.png', dpi=250)
    print(f'figure saved :: {name}_Chirp{i}_curve_vertical.png')
    plt.close()

for var, i in zip([slh_c1, slh_c2, slh_c3], (1, 2, 3)):
    plt.plot(h.lin2z(var), heights[f"C{i}Range"])
    plt.title(f"Mean Sensitivity limit for LIMRAD94 \n- Chirp {i} horizontal polarization -\n"
              f"Eurec4a - {begin_dt:%Y-%m-%d %H:%M} to {end_dt:%Y-%m-%d %H:%M} UTC\n"
              f"Chirptable: {chirptable}")
    plt.ylabel("Height [m]")
    plt.xlabel("Sensitivity Limit [dBZ]")
    plt.minorticks_on()
    plt.axes().xaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.axes().yaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.grid(True, which='both', axis='both', color="grey", linestyle='-', linewidth=1)
    plt.savefig(f'{name}_Chirp{i}_curve_horizontal.png', dpi=250)
    print(f'figure saved :: {name}_Chirp{i}_curve_horizontal.png')
    plt.close()
