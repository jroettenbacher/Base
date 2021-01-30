#!/usr/bin/env python
"""script to plot daily LWP of LIMRAD94 and LIMHAT, and their difference
input: via larda
output: ../plots/eurec4a_comp_LWP
author: Johannes Roettenbacher
"""

import sys
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('..')
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
campaign = "lindenberg_mol"
larda = pyLARDA.LARDA().connect(campaign)
begin = dt.datetime.strptime(larda.camp.VALID_DATES[0][0], "%Y%m%d")
end = dt.datetime.strptime(larda.camp.VALID_DATES[0][1], "%Y%m%d")
# set plotting options
outpath = f"../plots/{campaign}_comp_LWP"
y_lims = None

# set date
dates = pd.date_range(begin, end)
for begin_dt in dates:
    # begin_dt = dt.datetime(2020, 7, 15, 0, 0, 5)
    end_dt = begin_dt + dt.timedelta(0.9999)
    time_interval = [begin_dt, end_dt]

    radar_lwp = larda.read("LIMRAD94", "LWP", time_interval)
    hatpro_lwp = larda.read("HATPRO", "LWP", time_interval)
    hatpro_flag = larda.read("HATPRO", "flag", time_interval)
    hatpro_lwp['var'] = hatpro_lwp['var'] * 1000  # conversion to g/m2
    hatpro_lwp['var_unit'] = 'g m-2'

    # interpolate HATPRO data on radar time
    hatpro_lwp_ip = trans.interpolate1d(hatpro_lwp, new_time=radar_lwp['ts'])
    hatpro_lwp_ip['var'][np.isnan(hatpro_lwp_ip['var'])] = 0  # set nan to zero
    hatpro_lwp_ip['var'][np.isinf(hatpro_lwp_ip['var'])] = 0  # set inf to zero
    # check for HATPRO quality flags
    rainflag = hatpro_flag['var'] == 8  # rain flag
    # flag = hatpro_flag['var'] > 8  # any other flag
    if any(rainflag):
        # do not interpolate flags but rather chose closest point to radar time step
        hatpro_flag_ip = h.select_closest(hatpro_flag, radar_lwp['ts'])
        rainflag_ip = hatpro_flag_ip['var'] == 8  # create rain flag with radar time
        # mask flagged data, do not use masked array rainflag but the data of it
        hatpro_lwp_ip['var'] = np.ma.masked_where(rainflag_ip.data, hatpro_lwp_ip['var'])
        radar_lwp['var'] = np.ma.masked_where(rainflag_ip.data, radar_lwp['var'])
        # get position of flags for vertical lines in plot
        vlines = [h.ts_to_dt(t) for t in hatpro_flag_ip['ts'][rainflag_ip]]
    else:
        vlines = []

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

    # plot title
    title = f"Liquid Water Path Comparison LIMRAD94 vs. HATPRO {radar_lwp['paraminfo']['location']} {begin_dt:%Y-%m-%d}"

    # make plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=[10, 6])
    dot1, = ax1.plot(matplotlib.dates.date2num(h_pdata['dt'][:]), h_pdata['var'][:], '.',
                     markersize=1, alpha=h_pdata['alpha'], label="LWP HATPRO")
    dot2, = ax1.plot(matplotlib.dates.date2num(r_pdata['dt'][:]), r_pdata['var'][:], '.',
                     markersize=1, alpha=r_pdata['alpha'], label="LWP LIMRAD94")

    # check if positions for vertical lines are available
    if len(vlines) > 0:
        for x in matplotlib.dates.date2num(vlines[:-1]):
            ax1.axvline(x, alpha=0.1, color='red')
        # add the last line with label to add to legend
        vline = ax1.axvline(vlines[-1], alpha=0.1, color='red', label='HATPRO rain flag')

    # generate more visible legend entries
    hatpro_lgd, = plt.plot([], '.', markersize=5, label=dot1.get_label(), color=dot1.get_color())
    limrad_lgd, = plt.plot([], '.', markersize=5, label=dot2.get_label(), color=dot2.get_color())
    try:
        flag_lgd = Line2D([], [], label=vline.get_label(), color=vline.get_color())
        # add legends and aesthetics
        ax1.legend(handles=[hatpro_lgd, limrad_lgd, flag_lgd], bbox_to_anchor=(1.01, 1), loc='upper left')
    except NameError:
        logger.debug("No flagged data was found")
        ax1.legend(handles=[hatpro_lgd, limrad_lgd], bbox_to_anchor=(1.01, 1), loc='upper left')

    # format axes
    # set new y_limits through var_lims
    h_pdata['var_lims'] = y_lims if y_lims is not None else h_pdata['var_lims']
    ax1, _ = trans._format_axis(fig, ax1, dot1, h_pdata)
    ax1.grid()
    ax1.set_xlabel("")  # remove x label

    # plot difference between LIMRAD and HATPRO
    line1, = ax2.plot(matplotlib.dates.date2num(diff_pdata['dt'][:]), diff_pdata['var'][:], 'g',
                      label='Difference\nLIMRAD94 - HATPRO')
    # set new y_limits through var_lims
    if abs(diff_pdata['var_lims'][0]) > 500:
        diff_pdata['var_lims'] = [-500, 500]
    ax2, _ = trans._format_axis(fig, ax2, line1, diff_pdata)
    ax2.grid()
    ax2.legend(handles=[line1], bbox_to_anchor=(1.01, 1.0))

    fig.suptitle(title, size=16)
    fig.tight_layout()

    # define output name
    name_append = "" if y_lims is None else "_zoom"
    figname = f"{campaign}_LWP-comp_LIMRAD94-HATPRO_{begin_dt:%Y%m%d}{name_append}.png"
    plt.savefig(f"{outpath}/{figname}")
    plt.close()
    logger.info(f"Saved {figname}")
