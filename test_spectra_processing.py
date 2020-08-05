#!/bin/usr/env python
import sys, datetime, time

sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')

import pyLARDA
from larda.pyLARDA.SpectraProcessing import load_spectra_rpgfmcw94

larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
begin_dt = datetime.datetime(2020, 2, 3, 0, 0, 5)
end_dt = datetime.datetime(2020, 2, 3, 23, 59, 55)
timespan = [begin_dt, end_dt]
begin_dt_zoom = datetime.datetime(2020, 2, 3, 18, 0, 0)
end_dt_zoom = datetime.datetime(2020, 2, 3, 18, 20, 0)
timespan_zoom = [begin_dt_zoom, end_dt_zoom]
container = load_spectra_rpgfmcw94(larda, timespan,
                                   heave_correction=True)