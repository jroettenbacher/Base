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
import matplotlib.pyplot as plt

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
moments.update({'Vel_cor': moments['VEL'], 'heave_corr': moments['VEL']})
# overwrite var with corrected mean Doppler velocities and heave correction
moments['Vel_cor']['var'] = np.ma.masked_where(moments['Ze']['mask'], new_vel)
moments['heave_corr']['var'] = np.ma.masked_where(moments['Ze']['mask'], heave_corr)
print("Done with heave correction")

########################################################################################################################
# plotting
########################################################################################################################
plot_path = "/projekt1/remsens/work/jroettenbacher/plots/heave_correction"

mdv['var_lims'] = [-7, 7]
# uncorrected MDV
fig, _ = pyLARDA.Transformations.plot_timeheight(mdv, rg_converter=False, title=True)
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input'
fig_name = name + '_MDV_uncorrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# uncorrected MDV zoom
begin_dt = dt.datetime(2020, 2, 5, 9, 0, 0)
end_dt = dt.datetime(2020, 2, 5, 11, 30, 0)
fig, _ = pyLARDA.Transformations.plot_timeheight(mdv, rg_converter=False, title=True, time_interval=[begin_dt, end_dt])
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input'
fig_name = name + '_MDV_uncorrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

Vel_cor = moments['Vel_cor']
Vel_cor['var_lims'] = [-7, 7]
# corrected MDV
begin_dt = dt.datetime(2020, 2, 5, 0, 0, 5)
end_dt = dt.datetime(2020, 2, 5, 23, 59, 55)
fig, _ = pyLARDA.Transformations.plot_timeheight(Vel_cor, rg_converter=False, title=True)
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input'
fig_name = name + '_MDV_corrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# corrected MDV zoom
begin_dt = dt.datetime(2020, 2, 5, 9, 0, 0)
end_dt = dt.datetime(2020, 2, 5, 11, 30, 0)
fig, _ = pyLARDA.Transformations.plot_timeheight(Vel_cor, rg_converter=False, title=True,
                                                 time_interval=[begin_dt, end_dt])
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input'
fig_name = name + '_MDV_corrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

heave_corr = moments["heave_corr"]
heave_corr['var_lims'] = [-1.5, 1.5]
# heave correction
begin_dt = dt.datetime(2020, 2, 5, 0, 0, 5)
end_dt = dt.datetime(2020, 2, 5, 23, 59, 55)
fig, _ = pyLARDA.Transformations.plot_timeheight(heave_corr, rg_converter=False, title=True)
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input'
fig_name = name + '_heave_correction.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# heave correction zoom
begin_dt = dt.datetime(2020, 2, 5, 9, 0, 0)
end_dt = dt.datetime(2020, 2, 5, 11, 30, 0)
fig, _ = pyLARDA.Transformations.plot_timeheight(heave_corr, rg_converter=False, title=True,
                                                 time_interval=[begin_dt, end_dt])
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input'
fig_name = name + 'heave_correction.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# comparison of heave, pitch_heave and roll_heave
