#!/usr/bin/env python
"""Plot fft power spectra for each day, hour, chirp for both heave correction approaches
"""

import sys
sys.path.append("/projekt1/remsens/work/jroettenbacher/Base/larda")
sys.path.append('.')
import pyLARDA
import pyLARDA.helpers as h
import datetime
import functions_jr as jr
import numpy as np
from scipy.interpolate import CubicSpline
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# connect to campaign
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)

method_name, args, kwargs = h._method_info_from_argv(sys.argv)

# check argument/kwargs
if 'date' in kwargs:
    date = str(kwargs['date'])
    begin_dt = datetime.datetime.strptime(date + ' 00:00:05', '%Y%m%d %H:%M:%S')
    end_dt = datetime.datetime.strptime(date + ' 23:59:55', '%Y%m%d %H:%M:%S')
else:
    date = '20200120'
    begin_dt = datetime.datetime.strptime(date + ' 00:00:05', '%Y%m%d %H:%M:%S')
    end_dt = datetime.datetime.strptime(date + ' 23:59:55', '%Y%m%d %H:%M:%S')

range_ = [0, 'max']
TIME_SPAN_ = [begin_dt, end_dt]

seapath = jr.read_seapath(begin_dt)
seapath = jr.calc_heave_rate(seapath).dropna()
Cs_w_radar = CubicSpline(seapath.index, seapath['heave_rate_radar'])
mdv = larda.read("LIMRAD94", "VEL", TIME_SPAN_, range_)
mdv['var'][mdv['mask']] = np.nan
# load additional variables
mdv.update({var: larda.read("LIMRAD94", var, TIME_SPAN_, range_) for var in ['C1Range', 'C2Range', 'C3Range']})
mdv_cor = larda.read("LIMRAD94_cni_hc_jr", "Vel", TIME_SPAN_, range_)
mdv_cor['var'][mdv_cor['mask']] = np.nan
mdv_cor_roll = larda.read("LIMRAD94_cni_hc_jr", "Vel_roll", TIME_SPAN_, range_)
mdv_cor_roll['var'][mdv_cor_roll['mask']] = np.nan
rg_borders = jr.get_range_bin_borders(3, mdv)
rg_borders_id = rg_borders - np.array([0, 1, 1, 1])
chirp_ts = jr.calc_chirp_timestamps(mdv['ts'], begin_dt, version='start')
chirp_ts_shifted, _ = jr.calc_shifted_chirp_timestamps(mdv['ts'], mdv['var'], chirp_ts, rg_borders_id, 400, Cs_w_radar)

jr.plot_fft_spectra(mdv['var'], chirp_ts, mdv_cor['var'], chirp_ts_shifted, mdv_cor_roll['var'], 3, rg_borders_id, 400, seapath)
