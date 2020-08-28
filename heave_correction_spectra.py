#!/bin/python

# script for quality control of heave correction applied to spectra
########################################################################################################################
# library import
########################################################################################################################
import sys
import datetime as dt
import numpy as np
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')
import pyLARDA
import pyLARDA.helpers as h
from pyLARDA.SpectraProcessing import heave_correction, heave_correction_spectra
import matplotlib.pyplot as plt

# read in heave corrected cloudnet input file
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)

# define test cases
# begin_dt = dt.datetime(2020, 2, 5, 0, 0, 5)
# end_dt = dt.datetime(2020, 2, 5, 23, 59, 55)
# begin_dt_zoom = dt.datetime(2020, 2, 5, 9, 25, 0)
# end_dt_zoom = dt.datetime(2020, 2, 5, 9, 35, 0)
# begin_dt = dt.datetime(2020, 1, 28, 0, 0, 5)
# end_dt = dt.datetime(2020, 1, 28, 23, 59, 55)
# begin_dt_zoom = dt.datetime(2020, 1, 28, 17, 12, 0)
# end_dt_zoom = dt.datetime(2020, 1, 28, 17, 22, 0)
begin_dt = dt.datetime(2020, 2, 10, 0, 0, 5)
end_dt = dt.datetime(2020, 2, 10, 23, 59, 55)
begin_dt_zoom = dt.datetime(2020, 2, 10, 22, 10, 0)
end_dt_zoom = dt.datetime(2020, 2, 10, 22, 20, 0)
# begin_dt = dt.datetime(2020, 2, 2, 0, 0, 5)
# end_dt = dt.datetime(2020, 2, 2, 23, 59, 55)
# begin_dt_zoom = dt.datetime(2020, 2, 2, 11, 0, 0)
# end_dt_zoom = dt.datetime(2020, 2, 2, 11, 20, 0)
# begin_dt = dt.datetime(2020, 2, 3, 0, 0, 5)
# end_dt = dt.datetime(2020, 2, 3, 23, 59, 55)
# begin_dt_zoom = dt.datetime(2020, 2, 3, 18, 0, 0)
# end_dt_zoom = dt.datetime(2020, 2, 3, 18, 20, 0)

plot_range = [0, 'max']
# read in heave corrected velocity
mdv = larda.read("LIMRAD94_cni_hc", "Vel", [begin_dt, end_dt], plot_range)
mdv['var_lims'] = [-7, 7]
# read in corrected and dealiased velocity
mdv_2 = larda.read("LIMRAD94_cni_hc_dea", "Vel", [begin_dt, end_dt], plot_range)
mdv_2['var_lims'] = [-7, 7]
# read in uncorrected velocity from old cloudnet input file
mdv_uncorr = larda.read('LIMRAD94_cn_input', 'Vel', [begin_dt, end_dt], plot_range)
mdv_uncorr['var_lims'] = [-7, 7]
# read in uncorrected velocity from new cloudnet input files
mdv_uncorr_2 = larda.read('LIMRAD94_cni', 'Vel', [begin_dt, end_dt], plot_range)
mdv_uncorr_2['var_lims'] = [-7, 7]

# put every variable in a dict
moments = {'VEL_cor': mdv,
           'VEL_cor2': mdv_2,
           'VEL': mdv_uncorr, 'VEL2': mdv_uncorr_2,
           'VEL_uncor-cor': mdv,
           'VEL_uncor-cor2': mdv_2,
           'heave_corr': mdv, 'n_dopp_shift': mdv,
           'VEL_uncor-cor-heave_corr2': mdv_2,
           'VEL_uncor-cor-heave_corr': mdv}
for var in ['C1Range', 'C2Range', 'C3Range', 'SeqIntTime', 'Inc_ElA', 'MaxVel', 'DoppLen']:
    print('loading variable from LV1 :: ' + var)
    moments.update({var: larda.read('LIMRAD94', var, [begin_dt, end_dt], [0, 'max'])})
moments['SeqIntTime'] = moments['SeqIntTime']['var'][0]
moments['MaxVel'] = moments['MaxVel']['var'][0]
moments['DoppLen'] = moments['DoppLen']['var'][0]
# calculate heave correction array for corresponding day
new_vel, heave_cor_array, seapath_out = heave_correction(moments, begin_dt)
# calculate heave correction array with spectra processing
# trick to run it without spectra
moments['VHSpec'] = mdv
_, heave_cor_array_2, n_dopp_bins_shift, seapath_out_2 = heave_correction_spectra(moments, begin_dt)

# # check whether both heave correction functions return the same heave rate and correction array
# n_nans = np.sum(np.isnan(seapath_out["Heave Rate [m/s]"]))
# print(f'number of nans in heave rate: {n_nans}')
# print(f'heave rates are the same: {np.sum(seapath_out["Heave Rate [m/s]"] == seapath_out_2["Heave Rate [m/s]"]) == seapath_out.shape[0]-n_nans}')
# n_nans_2 = np.sum(np.isnan(heave_cor_array))
# print(f'number of nans in heave cor array: {n_nans_2}')
# print(f'heave cor arrays are the same: {(np.sum(heave_cor_array == heave_cor_array_2)) == heave_cor_array.size-n_nans_2}')
# # -> They do!

# updated mdv var with new values
moments['VEL_uncor-cor'] = h.put_in_container(mdv_uncorr['var'] - mdv['var'], moments['VEL_uncor-cor'])
moments['VEL_uncor-cor2'] = h.put_in_container(mdv_uncorr_2['var'] - mdv_2['var'], moments['VEL_uncor-cor2'])
moments['heave_corr'] = h.put_in_container(heave_cor_array, moments['heave_corr'])
moments['n_dopp_shift'] = h.put_in_container(heave_cor_array, moments['n_dopp_shift'])
moments['VEL_uncor-cor-heave_corr'] = h.put_in_container(moments['VEL_uncor-cor']['var'] - heave_cor_array, moments['VEL_uncor-cor-heave_corr'])
moments['VEL_uncor-cor-heave_corr2'] = h.put_in_container(moments['VEL_uncor-cor2']['var'] - heave_cor_array, moments['VEL_uncor-cor-heave_corr2'])
# rename variables
moments['VEL_uncor-cor']['name'] = 'VEL_uncor-cor'
moments['VEL_uncor-cor2']['name'] = 'VEL_uncor-cor2'
moments['VEL_uncor-cor-heave_corr']['name'] = 'VEL_uncor-cor-heave_corr'
moments['VEL_uncor-cor-heave_corr2']['name'] = 'VEL_uncor-cor-heave_corr2'
moments['heave_corr']['name'] = 'heave_corr'
moments['n_dopp_shift']['name'] = 'n_dopp_bin_shifts'

########################################################################################################################
# plotting
########################################################################################################################
plot_path = "/projekt1/remsens/work/jroettenbacher/plots/heave_correction_spectra"
plot_range = [0, 3000]
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km'
name_zoom = f'{plot_path}/{begin_dt_zoom:%Y%m%d_%H%M}_{end_dt_zoom:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km'

# uncorrected MDV zoom
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['VEL'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom],
                                                 range_interval=plot_range)
fig_name = f"{name_zoom}_{moments['VEL']['paraminfo']['system']}_MDV_uncorrected.png"
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# corrected MDV
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['VEL_cor'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt, end_dt],
                                                 range_interval=plot_range)
fig_name = f"{name}_{moments['VEL_cor']['paraminfo']['system']}_MDV_corrected.png"
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# corrected MDV zoom
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['VEL_cor'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom],
                                                 range_interval=plot_range)
fig_name = f"{name_zoom}_{moments['VEL_cor']['paraminfo']['system']}_MDV_corrected.png"
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# corrected MDV dealiased zoom
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['VEL_cor2'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom],
                                                 range_interval=plot_range)
fig_name = f"{name_zoom}_{moments['VEL_cor2']['paraminfo']['system']}_MDV_corrected.png"
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# corrected - uncorrected MDV zoom
moments['VEL_uncor-cor']['var_lims'] = [-1, 1]
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['VEL_uncor-cor'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom],
                                                 range_interval=plot_range)
fig_name = f"{name_zoom}_{moments['VEL_uncor-cor']['paraminfo']['system']}_MDV_uncorrected-corrected.png"
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# corrected - uncorrected MDV dealiased zoom
moments['VEL_uncor-cor2']['var_lims'] = [-1, 1]
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['VEL_uncor-cor2'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom],
                                                 range_interval=plot_range)
fig_name = f"{name_zoom}_{moments['VEL_uncor-cor2']['paraminfo']['system']}_MDV_uncorrected-corrected.png"
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# heave correction array zoom
moments['heave_corr']['var_lims'] = [-1, 1]
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['heave_corr'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom],
                                                 range_interval=plot_range)
fig_name = f"{name_zoom}_{moments['heave_corr']['paraminfo']['system']}_heave_corr.png"
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# doppler bin shift array zoom
moments['n_dopp_shift']['var_lims'] = [-3, 3]
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['n_dopp_shift'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom],
                                                 range_interval=plot_range)
fig_name = f"{name_zoom}_{moments['n_dopp_shift']['paraminfo']['system']}_n_dopp_shift.png"
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# difference between difference between Vel and Vel_cor and heave correction
moments['VEL_uncor-cor-heave_corr']['var_lims'] = [-1, 1]
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['VEL_uncor-cor-heave_corr'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom],
                                                 range_interval=plot_range)
fig_name = f"{name_zoom}_{moments['VEL_uncor-cor-heave_corr']['paraminfo']['system']}_MDV_uncor-cor-heave_corr.png"
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# difference between difference between Vel and Vel_cor dealiased and heave correction
moments['VEL_uncor-cor-heave_corr2']['var_lims'] = [-1, 1]
fig, _ = pyLARDA.Transformations.plot_timeheight(moments['VEL_uncor-cor-heave_corr2'], rg_converter=False, title=True,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom],
                                                 range_interval=plot_range)
fig_name = f"{name_zoom}_{moments['VEL_uncor-cor-heave_corr2']['paraminfo']['system']}_MDV_uncor-cor-heave_corr.png"
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# plot differently zoomed plots of corrected and uncorrected MDV (spectra) 25.01.2020
begin_dt = dt.datetime(2020,  2, 5, 0, 0, 0)
end_dt = dt.datetime(2020, 2, 5, 20, 0, 0)
plot_range = [0, 'max']
mdv = larda.read('LIMRAD94_cni_hc', 'Vel', [begin_dt, end_dt], plot_range)
mdv['var_lims'] = [-7, 7]
mdv_uncor = larda.read('LIMRAD94_cni', 'Vel', [begin_dt, end_dt], plot_range)
mdv_uncor['var_lims'] = [-7, 7]
hours = [12, 6, 3, 1, 0.5]  # different time range
max_ranges = [3000, 6000, 10000]  # different height range
# # testing
# hour = hours[-1]
# max_range = max_ranges[0]
for var, name in zip([mdv, mdv_uncor], ['spectra_corrected', 'uncorrected']):
    for hour in hours:
        for max_range in max_ranges:
            end_dt_plot = begin_dt + dt.timedelta(hours=hour)
            plot_range = [0, max_range]
            fig, _ = pyLARDA.Transformations.plot_timeheight(var, rg_converter=False, title=True,
                                                             time_interval=[begin_dt, end_dt_plot],
                                                             range_interval=plot_range)
            fig_name = f"{plot_path}/{begin_dt:%Y%m%d_%H%M}-{end_dt_plot:%Y%m%d_%H%M}_{max_range/1000:.0f}km_MDV_{name}.png"
            fig.savefig(fig_name, dpi=250)
            plt.close()
            print(f'figure saved :: {fig_name}')
