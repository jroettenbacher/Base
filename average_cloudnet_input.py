#!/usr/bin/env python
# script to average cloudnet input files to a certain time height resolution and select only relevant variables
# input cloudnet input netcdf files (from RV-Meteor)
# output averaged netcdf files

import sys
import time
import datetime as dt
sys.path.append(r'C:\Users\Johannes\PycharmProjects\Base\larda')
sys.path.append('.')
import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.NcWrite as nc
from larda.pyLARDA.spec2mom_limrad94 import spectra2moments, build_extended_container
import logging
import numpy as np

# Load LARDA
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
begin_dt = dt.date(2020, 1, 18)
end_dt = dt.date(2020, 2, 19)
# loop through all files in date range

