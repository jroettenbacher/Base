#!/bin/python

# script for quality control off heave correction
########################################################################################################################
# library import
########################################################################################################################
import sys, time
import datetime as dt
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import pyLARDA
import pyLARDA.helpers as h
import logging
import numpy as np
import functions_jr as jr

# run heave correction on cloudnet input file
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
begin_dt = dt.datetime(2020, 2, 5, 0, 0, 5)
end_dt = dt.datetime(2020, 2, 5, 23, 59, 55)
plot_range = [0, 'max']
mdv = larda.read("LIMRAD94_cn_input", "Vel", [begin_dt, end_dt], plot_range)
moments = {"VEL": mdv}
for var in ['MaxVel', 'DoppLen', 'C1Range', 'C2Range', 'C3Range']:
    print('loading variable from LV1 :: ' + var)
    moments.update({var: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'])})
new_vel, heave_corr, seapath_chirptimes, seapath_out = jr.heave_correction(moments, begin_dt)
print("Done with heave correction")



