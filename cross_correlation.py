#!/usr/bin/env python
"""Script to compute cross correlation between DSHIP data and LIMRAD94 data to check for possible time shift
input: DSHIP data, LIMRAD94 Doppler velocity
author: Johannes Roettenbacher
"""

import sys
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import datetime as dt
import logging
import functions_jr as jr
import numpy as np

log = logging.getLogger('__main__')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

dt = dt.datetime(2020, 2, 15, 0, 0, 0)
seapath = jr.read_seapath(dt)
versions = [1, 2]
for version in versions:
    t_shift, shift, seapath = jr.calc_time_shift_limrad_seapath(seapath, version)
    print(f"Time shift calculated from {dt:%Y-%m-%d}: {t_shift:.4f} with version {version}.")

dt = dt.datetime(2020, 2, 16, 0, 0, 0)
seapath = jr.read_seapath(dt)
for version in versions:
    t_shift, shift, seapath = jr.calc_time_shift_limrad_seapath(seapath, version)
    print(f"Time shift calculated from {dt:%Y-%m-%d}: {t_shift:.4f} with version {version}.")
