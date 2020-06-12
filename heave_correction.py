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
begin_dt = dt.datetime(2020, 2, 10, 0, 0, 5)
end_dt = dt.datetime(2020, 2, 10, 23, 59, 55)
plot_range = [0, 'max']
only_heave = False
use_cross_product = True
mdv = larda.read("LIMRAD94_cn_input", "Vel", [begin_dt, end_dt], plot_range)
moments = {"VEL": mdv}
for var in ['C1Range', 'C2Range', 'C3Range', 'SeqIntTime']:
    print('loading variable from LV1 :: ' + var)
    moments.update({var: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'])})
new_vel, heave_corr, seapath_chirptimes, seapath_out = jr.heave_correction(moments, begin_dt, only_heave=only_heave,
                                                                           use_cross_product=use_cross_product)
moments.update({'Vel_cor': moments['VEL'], 'heave_corr': moments['VEL'], 'Vel_cor-Vel': moments['VEL'],
                'Vel-Vel_cor': moments['VEL']})
# overwrite var with corrected mean Doppler velocities and heave correction
moments['Vel_cor'] = h.put_in_container(new_vel, moments['Vel_cor'])
moments['heave_corr'] = h.put_in_container(heave_corr, moments['heave_corr'])
moments['Vel_cor-Vel'] = h.put_in_container(new_vel - moments['VEL']['var'], moments['Vel_cor-Vel'])
moments['Vel-Vel_cor'] = h.put_in_container(moments['VEL']['var'] - new_vel, moments['Vel-Vel_cor'])
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
begin_dt_zoom = dt.datetime(2020, 2, 10, 22, 10, 0)
end_dt_zoom = dt.datetime(2020, 2, 10, 22, 20, 0)
name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input_meanHR'
name_zoom = f'{plot_path}/{begin_dt_zoom:%Y%m%d_%H%M}_{end_dt_zoom:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input_meanHR'
if use_cross_product:
    name = f'{plot_path}/{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input_meanHR_cross_product'
    name_zoom = f'{plot_path}/{begin_dt_zoom:%Y%m%d_%H%M}_{end_dt_zoom:%Y%m%d_%H%M}_{plot_range[1] / 1000:.0f}km_cloudnet_input_meanHR_cross_product'

mdv['var_lims'] = [-7, 7]
# # uncorrected MDV
# fig, _ = pyLARDA.Transformations.plot_timeheight(mdv, rg_converter=False, title=True, range_interval=plot_range)
# fig_name = name + '_MDV_uncorrected.png'
# fig.savefig(fig_name, dpi=250)
# plt.close()
# print(f'figure saved :: {fig_name}')

# uncorrected MDV zoom
fig, _ = pyLARDA.Transformations.plot_timeheight(mdv, rg_converter=False, title=True,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom],
                                                 range_interval=plot_range)
fig_name = name_zoom + '_MDV_uncorrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

Vel_cor = moments['Vel_cor']
Vel_cor['var_lims'] = [-7, 7]
# # corrected MDV
# fig, _ = pyLARDA.Transformations.plot_timeheight(Vel_cor, rg_converter=False, title=True, range_interval=plot_range)
# fig_name = name + '_MDV_corrected.png'
# fig.savefig(fig_name, dpi=250)
# plt.close()
# print(f'figure saved :: {fig_name}')

# corrected MDV zoom
fig, _ = pyLARDA.Transformations.plot_timeheight(Vel_cor, rg_converter=False, title=True, range_interval=plot_range,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom])
fig_name = name_zoom + '_MDV_corrected.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

cor_meas = moments['Vel_cor-Vel']
cor_meas['var_lims'] = [-1, 1]
# # corrected MDV - measured MDV
# fig, _ = pyLARDA.Transformations.plot_timeheight(cor_meas, rg_converter=False, title=True, range_interval=plot_range)
# fig_name = name + '_MDV_corrected-measured.png'
# fig.savefig(fig_name, dpi=250)
# plt.close()
# print(f'figure saved :: {fig_name}')

# corrected MDV - measured MDV zoom
fig, _ = pyLARDA.Transformations.plot_timeheight(cor_meas, rg_converter=False, title=True, range_interval=plot_range,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom])
fig_name = name_zoom + '_MDV_corrected-measured.png'
fig.savefig(fig_name, dpi=250)
plt.close()
print(f'figure saved :: {fig_name}')

# meas_cor = moments['Vel-Vel_cor']
# meas_cor['var_lims'] = [-1, 1]
# # measured MDV - corrected MDV
# fig, _ = pyLARDA.Transformations.plot_timeheight(meas_cor, rg_converter=False, title=True, range_interval=plot_range)
# fig_name = name + '_MDV_measured-corrected.png'
# fig.savefig(fig_name, dpi=250)
# plt.close()
# print(f'figure saved :: {fig_name}')
#
# # measured MDV - corrected MDV zoom
# fig, _ = pyLARDA.Transformations.plot_timeheight(meas_cor, rg_converter=False, title=True, range_interval=plot_range,
#                                                  time_interval=[begin_dt_zoom, end_dt_zoom])
# fig_name = name_zoom + '_MDV_measured-corrected.png'
# fig.savefig(fig_name, dpi=250)
# plt.close()
# print(f'figure saved :: {fig_name}')

heave_corr = moments["heave_corr"]
heave_corr['var_lims'] = [-1, 1]
# # heave correction
# fig, _ = pyLARDA.Transformations.plot_timeheight(heave_corr, rg_converter=False, title=True, range_interval=plot_range)
# fig_name = name + '_heave_correction.png'
# fig.savefig(fig_name, dpi=250)
# plt.close()
# print(f'figure saved :: {fig_name}')

# heave correction zoom
fig, _ = pyLARDA.Transformations.plot_timeheight(heave_corr, rg_converter=False, title=True, range_interval=plot_range,
                                                 time_interval=[begin_dt_zoom, end_dt_zoom])
fig_name = name_zoom + '_heave_correction.png'
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
seapath_out.columns = ("heading", "heave", "pitch", "roll", "radar_heave", "pitch_heave", "roll_heave", "heave_rate",
                       "chirp_no")
# decide on csv file name
if only_heave and use_cross_product:
    csv_name = f"seapath_out_{begin_dt:%Y%m%d}_only_heave_cross_product.csv"
elif only_heave and not use_cross_product:
    csv_name = f"seapath_out_{begin_dt:%Y%m%d}_only_heave.csv"
elif not only_heave and use_cross_product:
    csv_name = f"seapath_out_{begin_dt:%Y%m%d}_cross_product.csv"
else:
    csv_name = f"seapath_out_{begin_dt:%Y%m%d}.csv"
seapath_out.to_csv(f"{plot_path}/{csv_name}", encoding="1252", index_label="datetime")
