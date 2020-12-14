#!/usr/bin/env python
"""script to plot daily LWP of LIMRAD94 and LIMHAT, and their difference
input: via larda
output: ../plots/eurec4a_comp_LWP
author: Johannes Roettenbacher
"""

import sys
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import pyLARDA
import pyLARDA.Transformations as trans
import pyLARDA.helpers as h
import logging
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# connect campaign
larda = pyLARDA.LARDA().connect('eurec4a')

# set date
dates = pd.date_range(dt.date(2020, 1, 17), dt.date(2020, 2, 29))
for begin_dt in dates:
    # begin_dt = dt.datetime(2020, 2, 16)
    end_dt = begin_dt + dt.timedelta(1)
    time_interval = [begin_dt, end_dt]

    radar_lwp = larda.read("LIMRAD94", "LWP", time_interval)
    hatpro_lwp = larda.read("HATPRO", "LWP", time_interval)
    hatpro_flag = larda.read("HATPRO", "flag", time_interval)
    hatpro_lwp['var'] = hatpro_lwp['var'] * 1000  # conversion to g/m2
    hatpro_lwp['var_unit'] = 'g m-2'
    # use data of flag, ignore mask set by larda, necessary at days where no flag was set
    if isinstance(hatpro_flag['var'], np.ma.MaskedArray):
        hatpro_flag['var'] = np.copy(hatpro_flag['var'].data)
        hatpro_flag['mask'] = ~hatpro_flag['mask']

    # interpolate HATPRO data on radar time
    hatpro_lwp_ip = trans.interpolate1d(hatpro_lwp, new_time=radar_lwp['ts'])
    hatpro_flag_ip = trans.interpolate1d(hatpro_flag, new_time=radar_lwp['ts'])

    # mask flagged data
    hatpro_lwp_ip['var'] = np.ma.masked_where(hatpro_flag_ip['var'] > 8, hatpro_lwp_ip['var'])
    radar_lwp['var'] = np.ma.masked_where(hatpro_flag_ip['var'] > 8, radar_lwp['var'])

    # get difference
    diff_radar_hatpro = radar_lwp
    diff_radar_hatpro['var'] = radar_lwp['var'] - hatpro_lwp_ip['var']
    diff_radar_hatpro['var_lims'] = [diff_radar_hatpro['var'].min(), diff_radar_hatpro['var'].max()]

    # set up plotting data
    h_pdata = trans._copy_data(hatpro_lwp_ip)
    h_pdata['dt'], h_pdata['var'] = trans._masked_jumps(h_pdata)
    r_pdata = trans._copy_data(radar_lwp)
    r_pdata['dt'], r_pdata['var'] = trans._masked_jumps(r_pdata)
    diff_pdata = trans._copy_data(diff_radar_hatpro)
    diff_pdata['dt'], diff_pdata['var'] = trans._masked_jumps(diff_pdata)
    # get vertical lines at flagged locations
    vlines = [h.ts_to_dt(t) for t in h_pdata['ts'][hatpro_flag_ip['var'] > 8]]

    # plot title
    title = f"Liquid Water Path Comparison LIMRAD94 vs. HATPRO EUREC4A {begin_dt:%Y-%m-%d}"

    # make plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=[10, 6])
    line1 = ax1.plot(matplotlib.dates.date2num(h_pdata['dt'][:]), h_pdata['var'][:], '.',
                     markersize=1, alpha=h_pdata['alpha'], label="LWP HATPRO")
    ax1.plot(matplotlib.dates.date2num(r_pdata['dt'][:]), r_pdata['var'][:], '.',
             markersize=1, alpha=r_pdata['alpha'], label="LWP LIMRAD94")
    # check if a flag is available for this day
    if len(vlines) > 0:
        for x in matplotlib.dates.date2num(vlines[:-1]):
            ax1.axvline(x, alpha=0.5, color='red')
        # add the last line with label to add to legend
        ax1.axvline(vlines[-1], alpha=0.5, color='red', label='HATPRO flag')
        ax1.legend()
    ax1, _ = trans._format_axis(fig, ax1, line1, h_pdata)
    ax1.grid()
    ax1.set_xlabel("")
    line2 = ax2.plot(matplotlib.dates.date2num(diff_pdata['dt'][:]), diff_pdata['var'][:], 'g',
                     label='LIMRAD94 - HATPRO')
    ax2.legend()
    ax2, _ = trans._format_axis(fig, ax2, line2, diff_pdata)
    ax2.grid()
    fig.suptitle(title, size=16)
    figname = f"RV-Meteor_LWP-comp_LIMRAD94-HATPRO_{begin_dt:%Y%m%d}.png"
    plt.savefig(f"../plots/eurec4a_comp_LWP/{figname}")
    plt.close()
    logger.info(f"Saved {figname}")
