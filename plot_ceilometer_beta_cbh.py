#!/usr/bin/python

"""plot ceilometer beta_raw together with cloud base height"""

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
plot_path = "/projekt1/remsens/work/jroettenbacher/plots/ceilometer"
# Load LARDA
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
system = "CEILO"
begin_dt = dt.datetime(2020, 2, 5, 0, 0, 5)
end_dt = dt.datetime(2020, 2, 5, 23, 59, 55)
plot_range = [0, 'max']

beta_raw = larda.read(system, 'beta', [begin_dt, end_dt], plot_range)
cbh = larda.read(system, 'cbh', [begin_dt, end_dt])
time_list = cbh['ts']
var = cbh['var'].copy()
var = np.ma.masked_where(var < 100, var)

