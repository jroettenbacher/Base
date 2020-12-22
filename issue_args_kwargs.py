#!/usr/bin/env python

import sys
sys.path.append("/projekt1/remsens/work/jroettenbacher/Base/larda")
import pyLARDA
import datetime as dt
import logging
log = logging.getLogger('pyLARDA')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

larda = pyLARDA.LARDA().connect("eurec4a")

begin_dt = dt.datetime(2020, 1, 20, 0, 0, 0)
end_dt = begin_dt + dt.timedelta(hours=23, minutes=59, seconds=59)
time_interval = [begin_dt, end_dt]

# read in data
radar_ze = larda.read("LIMRAD94", "Ze", time_interval, [0, 'max'])
fig, ax = pyLARDA.Transformations.plot_timeheight2(radar_ze, range_interval=[0, 2000],
                                                   z_converter='lin2z',
                                                   rg_converter=False,
                                                   title=f"Issue title",
                                                   fontsize=16)
