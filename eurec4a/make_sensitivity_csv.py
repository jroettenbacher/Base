#!/usr/bin/env python
"""Script to write mean sensitivity limits from LIMRAD94 to a csv file
author: Johannes Roettenbacher"""

import sys
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')  # append path to larda
sys.path.append('.')
import pyLARDA
from pyLARDA.helpers import lin2z
import functions_jr as jr
import datetime as dt
import pandas as pd
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    log = logging.getLogger('__main__')
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())
    # leave out P06 because it has no values for the stepped plots and it only is one day of data
    out_path = "/projekt2/remsens/data_new/site-campaign/rv_meteor-eurec4a/instruments/LIMRAD94/sensitivity_curve"
    program = ['P07', 'P09']
    campaign = 'eurec4a'
    larda = pyLARDA.LARDA().connect(campaign)

    # get chirp table duration for file name
    begin_dts, end_dts = jr.get_chirp_table_durations(program)

    # get mean sensitivity limits
    stats_sl = jr.calc_sensitivity_curve(program, campaign, rain_flag=True)
    # get height bins for corresponding chirptable
    dummy_dates = {'P09': dt.datetime(2020, 1, 25, 0, 0, 5), 'P07': dt.datetime(2020, 2, 10, 0, 0, 5)}
    output = dict()  # dictionary for temporary data storage
    for p in program:
        height = larda.read("LIMRAD94", "Ze", [dummy_dates[p]], [0, 'max'])['rg']
        # write output to csv file
        output[p] = dict(height_m=height, mean_slh=stats_sl['mean_slh'][p], mean_slh_f=stats_sl['mean_slh_f'][p],
                         mean_slv=stats_sl['mean_slv'][p], mean_slv_f=stats_sl['mean_slv_f'][p],
                         median_slh=stats_sl['median_slh'][p], median_slh_f=stats_sl['median_slh_f'][p],
                         median_slv=stats_sl['median_slv'][p], median_slv_f=stats_sl['median_slv_f'][p])
        csv_name = f"{out_path}/RV-Meteor_cloudradar_sensitivity-limit_{begin_dts[p]:%Y%m%d}-{end_dts[p]:%Y%m%d}.csv"
        df = pd.DataFrame(output[p])
        df.to_csv(csv_name, index=False, sep=',')
        log.info(f"saved {csv_name}")
        # transform linear units to dBZ
        for var in df.columns[1:]:
            df[var] = lin2z(df[var].values)
        csv_name2 = f"{out_path}/RV-Meteor_cloudradar_sensitivity-limit_dBZ_{begin_dts[p]:%Y%m%d}-{end_dts[p]:%Y%m%d}.csv"
        df.to_csv(csv_name2, index=False, sep=',')
        log.info(f"saved {csv_name2}")
