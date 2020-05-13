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
import pyLARDA
import pyLARDA.helpers as h
import datetime as dt
import numpy as np
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
fig.savefig(name + '_allChirps.png', dpi=250)
print(f'figure saved :: {name}_allChirps.png')


