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
import pandas as pd

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
moments.update({'Vel_cor': moments['VEL'], 'heave_corr': moments['VEL'], 'Vel_cor-Vel': moments['VEL'],
                'Vel-Vel_cor': moments['VEL']})
# overwrite var with corrected mean Doppler velocities and heave correction
moments['Vel_cor']['var'] = np.ma.masked_where(moments['VEL']['mask'], new_vel)
moments['heave_corr']['var'] = np.ma.masked_where(moments['VEL']['mask'], heave_corr)
moments['Vel_cor-Vel']['var'] = np.ma.masked_where(moments['VEL']['mask'], new_vel - moments['VEL']['var'])
moments['Vel-Vel_cor']['var'] = np.ma.masked_where(moments['VEL']['mask'], moments['VEL']['var'] - new_vel)
moments['Vel_cor']['name'] = "Vel_cor"
moments['heave_corr']['name'] = "heave_corr"
moments['Vel_cor-Vel']['name'] = "Vel_cor-Vel"
moments['Vel-Vel_cor']['name'] = "Vel-Vel_cor"
print("Done with heave correction")

########################################################################################################################
# plotting
########################################################################################################################
plot_path = "/projekt1/remsens/work/jroettenbacher/plots/heave_correction"
plot_range = [0, 3000]
# mdv['var_lims'] = [-7, 7]
# uncorrected MDV
fig, _ = pyLARDA.Transformations.plot_timeheight(mdv, rg_converter=False, title=True, range_interval=plot_range)
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input'
fig_name = name + '_MDV_uncorrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# uncorrected MDV zoom
begin_dt = dt.datetime(2020, 2, 5, 9, 0, 0)
end_dt = dt.datetime(2020, 2, 5, 11, 0, 0)
fig, _ = pyLARDA.Transformations.plot_timeheight(mdv, rg_converter=False, title=True, time_interval=[begin_dt, end_dt],
                                                 range_interval=plot_range)
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input'
fig_name = name + '_MDV_uncorrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

Vel_cor = moments['Vel_cor']
# Vel_cor['var_lims'] = [-7, 7]
# corrected MDV
begin_dt = dt.datetime(2020, 2, 5, 0, 0, 5)
end_dt = dt.datetime(2020, 2, 5, 23, 59, 55)
fig, _ = pyLARDA.Transformations.plot_timeheight(Vel_cor, rg_converter=False, title=True, range_interval=plot_range)
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input'
fig_name = name + '_MDV_corrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# corrected MDV zoom
begin_dt = dt.datetime(2020, 2, 5, 9, 0, 0)
end_dt = dt.datetime(2020, 2, 5, 11, 0, 0)
fig, _ = pyLARDA.Transformations.plot_timeheight(Vel_cor, rg_converter=False, title=True, range_interval=plot_range,
                                                 time_interval=[begin_dt, end_dt])
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input'
fig_name = name + '_MDV_corrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

cor_meas = moments['Vel_cor-Vel']
# corrected MDV - measured MDV
begin_dt = dt.datetime(2020, 2, 5, 0, 0, 5)
end_dt = dt.datetime(2020, 2, 5, 23, 59, 55)
fig, _ = pyLARDA.Transformations.plot_timeheight(cor_meas, rg_converter=False, title=True, range_interval=plot_range)
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input'
fig_name = name + '_MDV_corrected-measured.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# corrected MDV - measured MDV zoom
begin_dt = dt.datetime(2020, 2, 5, 9, 0, 0)
end_dt = dt.datetime(2020, 2, 5, 11, 0, 0)
fig, _ = pyLARDA.Transformations.plot_timeheight(cor_meas, rg_converter=False, title=True, range_interval=plot_range,
                                                 time_interval=[begin_dt, end_dt])
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input'
fig_name = name + '_MDV_corrected-measured.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

meas_cor = moments['Vel-Vel_cor']
# corrected MDV - measured MDV
begin_dt = dt.datetime(2020, 2, 5, 0, 0, 5)
end_dt = dt.datetime(2020, 2, 5, 23, 59, 55)
fig, _ = pyLARDA.Transformations.plot_timeheight(meas_cor, rg_converter=False, title=True, range_interval=plot_range)
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input'
fig_name = name + '_MDV_measured-corrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# corrected MDV - measured MDV zoom
begin_dt = dt.datetime(2020, 2, 5, 9, 0, 0)
end_dt = dt.datetime(2020, 2, 5, 11, 0, 0)
fig, _ = pyLARDA.Transformations.plot_timeheight(meas_cor, rg_converter=False, title=True, range_interval=plot_range,
                                                 time_interval=[begin_dt, end_dt])
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input'
fig_name = name + '_MDV_measured-corrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

heave_corr = moments["heave_corr"]
heave_corr['var_lims'] = [-1, 1]
# heave correction
begin_dt = dt.datetime(2020, 2, 5, 0, 0, 5)
end_dt = dt.datetime(2020, 2, 5, 23, 59, 55)
fig, _ = pyLARDA.Transformations.plot_timeheight(heave_corr, rg_converter=False, title=True, range_interval=plot_range)
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input'
fig_name = name + '_heave_correction.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# heave correction zoom
begin_dt = dt.datetime(2020, 2, 5, 9, 0, 0)
end_dt = dt.datetime(2020, 2, 5, 11, 0, 0)
fig, _ = pyLARDA.Transformations.plot_timeheight(heave_corr, rg_converter=False, title=True, range_interval=plot_range,
                                                 time_interval=[begin_dt, end_dt])
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input'
fig_name = name + '_heave_correction.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# # comparison of heave, pitch_heave and roll_heave
# jr.set_presentation_plot_style()
# plt.style.use('default')
# df = seapath_out.loc[:, ["radar_heave", "Heave [m]", "pitch_heave", "roll_heave"]]
# fig, axs = plt.subplots(nrows=3)
# for i in range(3):
#     ax = axs[i]
#     df.loc[seapath_out['Chirp_no'] == i+1, :].plot(kind='line', ax=ax, legend=False)
#     # Shrink current axis by 20%
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#     ax.set_title(f"Chirp {i+1}")
#
# handles, labels = axs[1].get_legend_handles_labels()
# labels = ["Combined", "Ship", "Pitch induced", "Roll induced"]
# lgd = plt.legend(handles, labels, title="Heave Motions", bbox_to_anchor=(1.05, 1))
# fig.suptitle("Different heave motions influencing LIMRAD94")
# axs[1].set_ylabel("Heave [m]")
# axs[2].set_xlabel("Time")
# fig.autofmt_xdate()
# plt.savefig(f"{plot_path}/tmp.png", bbox_inches='tight')
# plt.close()
#
# # heave rate for each chirp
# df = seapath_out.loc[:, ["Heave Rate [m/s]"]]
# fig, axs = plt.subplots(nrows=3, sharey=True)
# for i in range(3):
#     ax = axs[i]
#     df.loc[seapath_out['Chirp_no'] == i+1, :].plot(kind='line', ax=ax, legend=False)
#     ax.set_title(f"Chirp {i+1}")
#
# fig.suptitle("Heave rate for each Chirp")
# axs[1].set_ylabel("Heave Rate [m/s]")
# axs[2].set_xlabel("Time")
# fig.autofmt_xdate()
# plt.tight_layout()
# plt.savefig(f"{plot_path}/tmp.png")
# plt.close()

# save seapath_out to csv for plotting with R
seapath_out.to_csv(f"{plot_path}/seapath_out.csv", encoding="1252", index_label="datetime")
