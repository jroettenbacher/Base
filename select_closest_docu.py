#!/usr/bin/env python
"""Script to show an example use of selecet_closest
author: Johannes Roettenbacher
"""
import sys
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
import pyLARDA
import pyLARDA.helpers as h
import datetime

# optionally configure the logging
# StreamHandler will print to console
import logging
log = logging.getLogger('pyLARDA')
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

# init larda
# either using local data
larda = pyLARDA.LARDA()
print("available campaigns ", larda.campaign_list)

# select a campaign
larda.connect('eurec4a')

begin_dt = datetime.datetime(2020, 2, 16, 0, 0, 5)
end_dt = datetime.datetime(2020, 2, 16, 23, 59, 55)

import matplotlib
radar_lwp = larda.read("LIMRAD94", "LWP", [begin_dt, end_dt])
hatpro_flag = larda.read("HATPRO", "flag", [begin_dt, end_dt])
# check for HATPRO quality flags
rainflag = hatpro_flag['var'] == 8  # rain flag
if any(rainflag):
    # do not interpolate flags but rather chose closest points to radar time steps
    hatpro_flag_ip = h.select_closest(hatpro_flag, radar_lwp['ts'])
    rainflag_ip = hatpro_flag_ip['var'] == 8  # create rain flag with radar time
    # get position of flags for vertical lines in plot
    vlines = [h.ts_to_dt(t) for t in hatpro_flag_ip['ts'][rainflag_ip]]
else:
    vlines = []

fig, ax = pyLARDA.Transformations.plot_timeseries(radar_lwp)
if len(vlines) > 0:
    for x in matplotlib.dates.date2num(vlines[:-1]):
        ax.axvline(x, alpha=0.1, color='red')
    # add the last line with label to add to legend
    vline = ax.axvline(vlines[-1], alpha=0.5, color='red', label='HATPRO rain flag')
ax.legend()
fig.savefig("radar-lwp_hatpro-rainflag.png")
