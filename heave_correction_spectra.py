#!/bin/python

# script for quality control of heave correction applied to spectra
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
import matplotlib.pyplot as plt
import pandas as pd

# read in heave corrected cloudnet input file
larda = pyLARDA.LARDA().connect('eurec4a')
begin_dt = dt.datetime(2020, 2, 5, 0, 0, 5)
end_dt = dt.datetime(2020, 2, 5, 23, 59, 55)
begin_dt_zoom = dt.datetime(2020, 2, 5, 9, 25, 0)
end_dt_zoom = dt.datetime(2020, 2, 5, 9, 35, 0)
# begin_dt = dt.datetime(2020, 1, 28, 0, 0, 5)
# end_dt = dt.datetime(2020, 1, 28, 23, 59, 55)
# begin_dt_zoom = dt.datetime(2020, 1, 28, 17, 12, 0)
# end_dt_zoom = dt.datetime(2020, 1, 28, 17, 22, 0)
# begin_dt = dt.datetime(2020, 2, 10, 0, 0, 5)
# end_dt = dt.datetime(2020, 2, 10, 23, 59, 55)
# begin_dt_zoom = dt.datetime(2020, 2, 10, 22, 10, 0)
# end_dt_zoom = dt.datetime(2020, 2, 10, 22, 20, 0)
# begin_dt = dt.datetime(2020, 2, 2, 0, 0, 5)
# end_dt = dt.datetime(2020, 2, 2, 23, 59, 55)
# begin_dt_zoom = dt.datetime(2020, 2, 2, 11, 0, 0)
# end_dt_zoom = dt.datetime(2020, 2, 2, 11, 20, 0)
# begin_dt = dt.datetime(2020, 2, 3, 0, 0, 5)
# end_dt = dt.datetime(2020, 2, 3, 23, 59, 55)
# begin_dt_zoom = dt.datetime(2020, 2, 3, 18, 0, 0)
# end_dt_zoom = dt.datetime(2020, 2, 3, 18, 20, 0)
plot_range = [0, 'max']
mdv = larda.read("LIMRAD94_cni_hc", "Vel", [begin_dt, end_dt], plot_range)
mdv['var_lims'] = [-7, 7]
mdv_uncorr = larda.read("LIMRAD94_cn_input", "Vel", [begin_dt, end_dt], plot_range)
mdv_uncorr['var_lims'] = [-7, 7]
moments = {"VEL_cor": mdv, "VEL": mdv_uncorr}

########################################################################################################################
# plotting
########################################################################################################################
plot_path = "/projekt1/remsens/work/jroettenbacher/plots/heave_correction_spectra"
plot_range = [0, 3000]
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cni_hc'
name_zoom = f'{plot_path}/{begin_dt_zoom:%Y%m%d_%H%M}_{end_dt_zoom:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cni_hc'

# uncorrected MDV zoom
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['mdv_uncorr'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom],
                                                 range_interval=plot_range)
fig_name = name_zoom + '_MDV_uncorrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# corrected MDV zoom
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['VEL'], rg_converter=False, title=True,
                                                 range_interval=plot_range, time_interval=[begin_dt_zoom, end_dt_zoom])
fig_name = name_zoom + '_MDV_corrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')
