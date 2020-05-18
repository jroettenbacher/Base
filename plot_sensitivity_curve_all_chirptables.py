#!/usr/bin/python

"""plot sensitivity curve for each chirp of LIMRAD94 for all chirptables
sensitivity limit is calculated for horizontal and vertical polarization
for more information see LIMRAD94 manual chapter 2.6
filter sensitivity during rain
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

# read in slv, slh for all 3 chirptables
# read in reflectivity and use mask to filter for rain
# take only non rainy profiles (no signal in 2:6 lowest range gates)
# 3 subplots each for a chirptable with mean sensitivity limit v and h

begin_dts = [dt.datetime(2020, 1, 17, 0, 0, 5), dt.datetime(2020, 1, 30, 15, 30, 5),
             dt.datetime(2020, 1, 31, 22, 30, 5)]
end_dts = [dt.datetime(2020, 1, 27, 0, 0, 5), dt.datetime(2020, 1, 30, 23, 42, 00),
           dt.datetime(2020, 2, 19, 23, 59, 55)]
plot_range = [0, 'max']
# names and program numbers of chirp tables
chirptables = ("tradewindCU (P09)", "Cu_small_Tint (P06)", "Cu_small_Tint2 (P07)")
programs = ["P09", "P06", "P07"]
name = f'{plot_path}/RV-Meteor_LIMRAD94_sensitivity_curves_all_chirptables.png'

# read in sensitivity variables over whole range (all chirps) and reflectivity from cloudnet input
slv = {}
slh = {}
radar_z = {}
for begin_dt, end_dt, program in zip(begin_dts, end_dts, programs):
    slv[program] = larda.read(system, "SLv", [begin_dt, end_dt], plot_range)
    slh[program] = larda.read(system, "SLh", [begin_dt, end_dt], plot_range)
    # radar_z[program] = larda.read("LIMRAD94_cn_input", "Ze", [begin_dt, end_dt], plot_range)

mean_slv = {}
mean_slh = {}
for program in programs:
    mean_slv[program] = np.mean(slv[program]['var'], axis=0)
    mean_slh[program] = np.mean(slh[program]['var'], axis=0)

########################################################################################################################
# PLOTTING
########################################################################################################################
plt.style.use('default')
plt.rcParams.update({'figure.figsize': (16, 9)})

# plot with all chirps and both polarizations
fig, axs = plt.subplots(ncols=3, constrained_layout=True, sharey='all', sharex='all')
for ax, program, i in zip(axs, programs, range(len(chirptables))):
    ax.plot(h.lin2z(mean_slv[program]), slv[program]['rg'], label='vertical')
    ax.plot(h.lin2z(mean_slh[program]), slh[program]['rg'], label='horizontal')
    ax.set_title(chirptables[i])
    ax.set_ylabel("Height [m]")
    ax.set_xlabel("Sensitivity Limit [dBZ]")
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.grid(True, which='both', axis='both', color="grey", linestyle='-', linewidth=1)
    ax.legend(title="Polarization")
fig.suptitle(f"Mean Sensitivity limit for LIMRAD94 \n"
             f"Eurec4a - whole duration of chirptable use", fontsize=16)
fig.savefig(f'{name}', dpi=250)
print(f'figure saved :: {name}')
plt.close()
