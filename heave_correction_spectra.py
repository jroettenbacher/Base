#!/bin/python

# script for quality control of heave correction applied to spectra
########################################################################################################################
# library import
########################################################################################################################
import sys
import datetime as dt
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import pyLARDA
import pyLARDA.helpers as h
from pyLARDA.SpectraProcessing import heave_correction
import matplotlib.pyplot as plt

# read in heave corrected cloudnet input file
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)
system = "LIMRAD94_cni_hc"
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
mdv = larda.read(system, "Vel", [begin_dt, end_dt], plot_range)
mdv['var_lims'] = [-7, 7]
mdv_uncorr = larda.read('LIMRAD94_cn_input', 'Vel', [begin_dt, end_dt], plot_range)
mdv_uncorr['var_lims'] = [-7, 7]
moments = {'VEL_cor': mdv, 'VEL': mdv_uncorr, 'VEL_cor-uncor': mdv, 'heave_corr': mdv, 'VEL_cor-uncor-heave_corr': mdv}
for var in ['C1Range', 'C2Range', 'C3Range', 'SeqIntTime', 'Inc_ElA']:
    print('loading variable from LV1 :: ' + var)
    moments.update({var: larda.read('LIMRAD94', var, [begin_dt, end_dt], [0, 'max'])})
moments['SeqIntTime'] = moments['SeqIntTime']['var'][0]
# calculate heave correction array for corresponding day
new_vel, heave_cor_array, seapath_out = heave_correction(moments, begin_dt)

# updated mdv var with new values
moments['VEL_cor-uncor'] = h.put_in_container(mdv['var'] - mdv_uncorr['var'], moments['VEL_cor-uncor'])
moments['heave_corr'] = h.put_in_container(heave_cor_array, moments['heave_corr'])
moments['VEL_cor-uncor-heave_corr'] = h.put_in_container(moments['VEL_cor-uncor']['var'] - heave_cor_array, moments['VEL_cor-uncor-heave_corr'])
# rename variables
moments['VEL_cor-uncor']['name'] = 'VEL_cor-uncor'
moments['heave_corr']['name'] = 'heave_corr'
moments['VEL_cor-uncor-heave_corr']['name'] = 'VEL_cor-uncor-heave_corr'

########################################################################################################################
# plotting
########################################################################################################################
plot_path = "/projekt1/remsens/work/jroettenbacher/plots/heave_correction_spectra"
plot_range = [0, 3000]
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cni_hc'
name_zoom = f'{plot_path}/{begin_dt_zoom:%Y%m%d_%H%M}_{end_dt_zoom:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cni_hc'

# uncorrected MDV zoom
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['VEL'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom],
                                                 range_interval=plot_range)
fig_name = name_zoom + '_MDV_uncorrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# corrected MDV
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['VEL_cor'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt, end_dt],
                                                 range_interval=plot_range)
fig_name = name + '_MDV_corrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# corrected MDV zoom
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['VEL_cor'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom],
                                                 range_interval=plot_range)
fig_name = name_zoom + '_MDV_corrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# corrected - uncorrected MDV zoom
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['VEL_cor-uncor'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom],
                                                 range_interval=plot_range)
fig_name = name_zoom + '_MDV_corrected-uncorrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# heave correction array zoom
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['heave_corr'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom],
                                                 range_interval=plot_range)
fig_name = name_zoom + '_heave_corr.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# difference between difference between Vel_cor and Vel and heave correction
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['VEL_cor-uncor-heave_corr'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom],
                                                 range_interval=plot_range)
fig_name = name_zoom + 'MDV_cor-uncor-heave_corr.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')
