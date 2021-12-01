#!/usr/bin/python3

import sys

# just needed to find pyLARDA from this location
sys.path.append('/projekt1/remsens/work/jroettenbacher/Base/larda')
sys.path.append('.')

import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pyLARDA
import pyLARDA.helpers as h
import datetime
import numpy as np
import pandas as pd

import logging

log = logging.getLogger('pyLARDA')
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler())

# define plot path
# plot_path = "/projekt2/remsens/data/campaigns/eurec4a/LIMRAD94/quicklooks/MDV_cor"
plot_path = "../plots"
# Load LARDA
larda = pyLARDA.LARDA().connect('eurec4a', build_lists=True)

# gather command line arguments
method_name, args, kwargs = h._method_info_from_argv(sys.argv)
# gather argument date
if 'date_begin' in kwargs:
    date_begin = str(kwargs['date_begin'])
    begin_dt = datetime.datetime.strptime(date_begin + ' 00:00:05', '%Y%m%d %H:%M:%S')
else:
    begin_dt = datetime.datetime(2020, 2, 18, 0, 0, 0)

if 'date_end' in kwargs:
    date_end = str(kwargs['date_end'])
    end_dt = datetime.datetime.strptime(date_end + ' 23:59:55', '%Y%m%d %H:%M:%S')
else:
    end_dt = datetime.datetime(2020, 2, 18, 23, 59, 55)

if 'plot_range' in kwargs:
    plot_range = [0, int(kwargs['plot_range'])]
else:
    plot_range = [0, 4000]

# read in moments
system = "LIMRAD94_cni_hc_ca"
radar_Z = larda.read(system, "Ze", [begin_dt, end_dt], plot_range)
# mask values = -999
radar_Z["var"] = np.ma.masked_where(radar_Z["var"] == -999, radar_Z["var"])
# overwrite mask in larda container -> does not change plot output
radar_Z["mask"] = radar_Z["var"].mask

radar_MDV = larda.read(system, "Vel", [begin_dt, end_dt], plot_range)
# overwrite mask in larda container
radar_MDV["mask"] = radar_Z["var"].mask
# radar_MDV["var_lims"] = [-7, 7]

# radar_MDV_cor = larda.read(system, "Vel_cor", [begin_dt, end_dt], plot_range)
# # overwrite mask in larda container
# radar_MDV_cor["mask"] = radar_Z["var"].mask
# # radar_MDV_cor["var_lims"] = [-7, 7]

# heave_corr = larda.read(system, "heave_corr", [begin_dt, end_dt], plot_range)

radar_sw = larda.read(system, "sw", [begin_dt, end_dt], plot_range)
# overwrite mask in larda container
radar_sw["mask"] = radar_Z["var"].mask
# radar_vel = larda.read(system, "vm", [begin_dt, end_dt], plot_range)
# radar_Inc_El = larda.read(system, "Inc_El", [begin_dt, end_dt])
# radar_Inc_ElA = larda.read(system, "Inc_ElA", [begin_dt, end_dt])
# radar_LDR = larda.read(system, "ldr", [begin_dt, end_dt], plot_range)
# cloud_bases_tops = larda.read(system, "cloud_bases_tops", [begin_dt, end_dt], plot_range)
# dt_list = np.asarray([datetime.datetime.utcfromtimestamp(time) for time in cloud_bases_tops['ts']])
# range_list = cloud_bases_tops['rg']/1000.0
# var = cloud_bases_tops['var']
# # mask -999 = fill value
# var = np.ma.masked_where(var == -999, var)

name = f'{plot_path}/' \
       f'{begin_dt:%Y%m%d_%H%M}_{end_dt:%Y%m%d_%H%M}_hcor_{plot_range[1] / 1000:.0f}km_cloudnet_input'

# plot Roll and pitch
# fig, ax = pyLARDA.Transformations.plot_timeseries(radar_Inc_El)
# formatted_datetime = (h.ts_to_dt(radar_Inc_El['ts'][0])).strftime("%Y-%m-%d")
# if not (h.ts_to_dt(radar_Inc_El['ts'][0])).strftime("%d") == (h.ts_to_dt(radar_Inc_El['ts'][-1])).strftime("%d"):
#     formatted_datetime = formatted_datetime + '-' + (h.ts_to_dt(radar_Inc_El['ts'][-1])).strftime("%d")
# ax.set_title(radar_Inc_El['paraminfo']['location'] + ', ' + formatted_datetime, fontsize=20)
# fig.savefig(f"{name}_Inc_El.png", dpi=250)

# radar_Z['var_unit'] = 'dBZ'
# radar_Z['colormap'] = 'jet'
fig, axs = plt.subplots(nrows=3, figsize=[14, 5.7])
fig1, axs[0] = pyLARDA.Transformations.plot_timeheight2(radar_Z, ax=axs[0], fig=fig, label="Radar Reflectivity Factor (dBZ)")
fig2, axs[1] = pyLARDA.Transformations.plot_timeheight2(radar_MDV, ax=axs[1], fig=fig, label="Heave Corrected Mean Doppler Velocity $(m\,s^{-1})$")
fig3, axs[2] = pyLARDA.Transformations.plot_timeheight2(radar_sw, ax=axs[2], fig=fig, label="Spectral Width $(m\,s^{-1}$")
plt.savefig(name + '_Javier.png', dpi=250)
print(f'figure saved :: {name}_Z.png')
plt.close()
#
# fig, ax = pyLARDA.Transformations.plot_timeheight(radar_Z, range_interval=plot_range, rg_converter=True, title=True,
#                                                   z_converter='lin2z')
# # create a binary color map for bases and tops
# cmap = ListedColormap(["darkgreen", "darkmagenta"])
# ax.pcolormesh(matplotlib.dates.date2num(dt_list[:]), range_list[:], np.transpose(var[:, :]),
#               label='cloud bases and tops', cmap=cmap)
# legend_elements = [Patch(facecolor='darkgreen', label='Bases'),
#                    Patch(facecolor='darkmagenta', label='Tops')]
# ax.legend(handles=legend_elements, title="Hydrometeor")
# fig.savefig(name + '_cbt_Z.png', dpi=250)
# print(f'figure saved :: {name}_cbt_Z.png')

fig, ax = pyLARDA.Transformations.plot_timeheight(radar_MDV, rg_converter=False, title=True)
fig.savefig(name + '_MDV.png', dpi=250)
plt.close()
print(f'figure saved :: {name}_MDV.png')

fig, ax = pyLARDA.Transformations.plot_timeheight(radar_MDV_cor, rg_converter=False, title=True)
fig.savefig(name + '_MDV_cor.png', dpi=250)
plt.close()
print(f'figure saved :: {name}_MDV_cor.png')

# plot difference between corrected and uncorrected Doppler velocity
radar_MDV_cor['var'] = radar_MDV_cor['var'] - radar_MDV['var']
radar_MDV_cor['var_lims'] = [-1.5, 1.5]
fig, ax = pyLARDA.Transformations.plot_timeheight(radar_MDV_cor, rg_converter=False, title=True)
fig.savefig(name + '_MDV_cor-MDV.png', dpi=250)
plt.close()
print(f'figure saved :: {name}_MDV_cor-MDV.png')

# plot heave rate for correcting mean Doppler velocity
fig, ax = pyLARDA.Transformations.plot_timeheight(heave_corr, rg_converter=False, title=True)
fig.savefig(name + '_heave_corr.png', dpi=250)
plt.close()
print(f'figure saved :: {name}_heave_corr.png')
# #
# fig, _ = pyLARDA.Transformations.plot_timeheight(radar_sw, rg_converter=True, title=True)
# fig.savefig(name + '_width.png', dpi=250)
# print(f'figure saved :: {name}_width.png')
#
# radar_LDR['var_lims'] = [-30, 0]
# radar_LDR['colormap'] = 'jet'
# fig, _ = pyLARDA.Transformations.plot_timeheight(radar_LDR, rg_converter=True, title=True)
# fig.savefig(name+'_LDR.png', dpi=250)
# print(f'figure saved :: {name}_LDR.png')
#
# radar_ZDR['var_lims'] = [-0.2, 1]
# radar_ZDR['colormap'] = 'jet'
# fig, _ = pyLARDA.Transformations.plot_timeheight(radar_ZDR, rg_converter=True, title=True)
# fig.savefig(name+'_ZDR.png', dpi=250)
# print(f'figure saved :: {name}_ZDR.png')
#
# radar_RHV['var_lims'] = [0.85, 1]
# radar_RHV['colormap'] = 'jet'
# fig, _ = pyLARDA.Transformations.plot_timeheight(radar_RHV, rg_converter=True, title=True)
# fig.savefig(name+'_RHV.png', dpi=250)
# print(f'figure saved :: {name}_RHV.png')
#
# radar_PhiDP['var_lims'] = [0, 5]
# radar_PhiDP['colormap'] = 'jet'
# fig, _ = pyLARDA.Transformations.plot_timeheight(radar_PhiDP, rg_converter=True, title=True)
# fig.savefig(name+'_PhiDP.png', dpi=250)
# print(f'figure saved :: {name}_PhiDP.png')

def remsens_limrad_quicklooks(container_dict, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import LogFormatter
    import matplotlib.colors as mcolors
    import time

    tstart = time.time()
    print('Plotting data...')

    time_list = container_dict['Ze']['ts']
    dt_list = [datetime.datetime.utcfromtimestamp(time) for time in time_list]

    if 'timespan' in kwargs and kwargs['timespan'] == '24h':
        dt_lim_left = datetime.datetime(dt_list[0].year, dt_list[0].month, dt_list[0].day, 0, 0)
        dt_lim_right = datetime.datetime(dt_list[0].year, dt_list[0].month, dt_list[0].day, 0, 0) + datetime.timedelta(days=1)
    else:
        dt_lim_left = dt_list[0]
        dt_lim_right = dt_list[-1]

    range_list = container_dict['Ze']['rg'] * 1.e-3  # convert to km
    ze = h.lin2z(container_dict['Ze']['var']).copy().T
    mdv = container_dict['VEL']['var'].copy().T
    sw = container_dict['sw']['var'].copy().T

    plot_range = kwargs['plot_range'] if 'plot_range' in kwargs else [0, 12.0]

    # create figure

    fig, ax = plt.subplots(3, figsize=(13, 16))

    # reflectivity plot
    ax[0].text(.015, .87, 'Radar reflectivity factor', horizontalalignment='left',
               transform=ax[0].transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.75))
    cp = ax[0].pcolormesh(dt_list, range_list, ze,
                          vmin=container_dict['Ze']['var_lims'][0],
                          vmax=container_dict['Ze']['var_lims'][1],
                          cmap=container_dict['Ze']['colormap'])
    divider = make_axes_locatable(ax[0])
    cax0 = divider.append_axes("right", size="3%", pad=0.3)
    cbar = fig.colorbar(cp, cax=cax0, ax=ax[0])
    cbar.set_label('dBZ')
    ax[0].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    print('Plotting data... Ze')

    # mean doppler velocity plot
    ax[1].text(.015, .87, 'Mean Doppler velocity', horizontalalignment='left', transform=ax[1].transAxes,
               fontsize=14, bbox=dict(facecolor='white', alpha=0.75))
    cp = ax[1].pcolormesh(dt_list, range_list, mdv,
                          vmin=container_dict['VEL']['var_lims'][0],
                          vmax=container_dict['VEL']['var_lims'][1],
                          cmap=container_dict['VEL']['colormap'])
    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("right", size="3%", pad=0.3)
    cbar = fig.colorbar(cp, cax=cax2, ax=ax[1])
    cbar.set_label('m/s')
    ax[1].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    print('Plotting data... mdv')

    # spectral width plot
    ax[2].text(.015, .87, 'Spectral width', horizontalalignment='left', transform=ax[2].transAxes,
               fontsize=14, bbox=dict(facecolor='white', alpha=0.75))
    cp = ax[2].pcolormesh(dt_list, range_list, sw,
                          norm=mcolors.LogNorm(vmin=0.1,
                                               vmax=container_dict['sw']['var_lims'][1]),
                          cmap=container_dict['sw']['colormap'])
    divider3 = make_axes_locatable(ax[2])
    cax3 = divider3.append_axes("right", size="3%", pad=0.3)
    formatter = LogFormatter(10, labelOnlyBase=False)
    cbar = fig.colorbar(cp, cax=cax3, ax=ax[2], format=formatter, ticks=[0.1, 0.2, 0.5, 1, 2])
    cbar.set_ticklabels([0.1, 0.2, 0.5, 1, 2])
    cbar.set_label('m/s')
    ax[2].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
    print('Plotting data... sw')

    yticks = np.arange(plot_range[0] / 1000., plot_range[1] / 1000. + 1, 2)  # y-axis ticks

    for iax in range(3):
        ax[iax].grid(linestyle=':')
        ax[iax].set_yticks(yticks)
        ax[iax].axes.tick_params(axis='both', direction='inout', length=10, width=1.5)
        ax[iax].set_ylabel('Height (km)', fontsize=14)
        ax[iax].set_xlim(left=dt_lim_left, right=dt_lim_right)
        ax[iax].set_ylim(top=plot_range[1] / 1000., bottom=plot_range[0] / 1000.)
        ax[iax].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

    fig.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0.20)

    print('plotting done, elapsed time = {:.3f} sec.'.format(time.time() - tstart))

    return fig, ax

container = dict(Ze=radar_Z, VEL=radar_MDV, sw=radar_sw)
remsens_limrad_quicklooks(container)