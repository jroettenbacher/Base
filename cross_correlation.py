#!/usr/bin/env python
"""Script to compute cross correlation between DSHIP data and LIMRAD94 data to check for possible time shift
input: DSHIP data
output: print to console of time shifts detected on given days and with used version
author: Johannes Roettenbacher
"""

import sys
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import datetime as dt
import logging
import functions_jr as jr

log = logging.getLogger('__main__')
log.setLevel(logging.ERROR)
log.addHandler(logging.StreamHandler())

date = dt.datetime(2020, 2, 15, 0, 0, 0)
seapath = jr.read_seapath(date)
seapath = jr.calc_heave_rate(seapath)
versions = [1, 2]
for version in versions:
    t_shift, shift, seapath = jr.calc_time_shift_limrad_seapath(seapath, version, plot_xcorr=True)
    print(f"Time shift calculated from {date:%Y-%m-%d}: {t_shift:.4f} with version {version}.")

date = dt.datetime(2020, 2, 16, 0, 0, 0)
seapath = jr.read_seapath(date)
seapath = jr.calc_heave_rate(seapath)
for version in versions:
    t_shift, shift, seapath = jr.calc_time_shift_limrad_seapath(seapath, version)
    print(f"Time shift calculated from {date:%Y-%m-%d}: {t_shift:.4f} with version {version}.")
