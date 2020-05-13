#!/usr/bin/python

"""plot ceilometer beta_raw together with cloud base height"""

import sys
# just needed to find pyLARDA from this location
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
# plot_path = "/projekt1/remsens/work/jroettenbacher/plots/ceilometer"
plot_path = "/projekt2/remsens/data/campaigns/eurec4a/RV-METEOR_CEILOMETER/quicklooks"
# Load LARDA
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
system = "CEILO"
for date in pd.date_range(dt.datetime(2020, 1, 17), dt.datetime(2020, 2, 19), freq='D'):
    # begin_dt = dt.datetime(2020, 2, 16, 0, 0, 5)
    # end_dt = dt.datetime(2020, 2, 16, 23, 59, 55)
    begin_dt = date + dt.timedelta(0, 5)
    end_dt = date + dt.timedelta(0, 23*60*60+59*60+55)
    plot_range = [0, 'max']

    # read in beta_raw
    beta_raw = larda.read(system, 'beta', [begin_dt, end_dt], plot_range)
    # beta_raw['var'] = np.ma.masked_where(beta_raw['var'] < 0, beta_raw['var'])

    cbh = larda.read(system, 'cbh', [begin_dt, end_dt])
    time_list = cbh['ts']
    dt_list = np.asarray([dt.datetime.utcfromtimestamp(time) for time in time_list])
    cbh_var = cbh['var'].copy()
    cbh_var = np.ma.masked_where(cbh_var < 100, cbh_var)

    if plot_range[1] == 'max':
        # name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_ceilo'
        name = f"{plot_path}(RV-Meteor_ceilometer_beta-raw+cbh_{begin_dt:%Y%m%d_%H%M}-{end_dt:%Y%m%d_%H%M}.png"
    else:
        # name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_ceilo'
        name = f"{plot_path}(RV-Meteor_ceilometer_beta-raw+cbh" \
               f"_{plot_range[1] / 1000:.0f}km_{begin_dt:%Y%m%d_%H%M}-{end_dt:%Y%m%d_%H%M}.png"

    beta_raw['name'] = 'beta raw'
    fig, ax = pyLARDA.Transformations.plot_timeheight(beta_raw, rg_converter=False, title=True)
    ax.plot(dt_list, cbh_var, '.', ms=1.5, color='purple', alpha=0.7)
    dot = mlines.Line2D([], [], ls='None', marker='o', color='purple', label='cloud base height ceilometer')
    ax.legend(handles=[dot], loc='upper right')
    fig_name = f'{name}_beta_raw+cbh.png'
    fig.savefig(fig_name, dpi=250)
    plt.close()
    print(f'figure saved :: {fig_name}')
